"""
Medical LangChain Agent for Revival Hospital System
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from tools.health_progress_tool import HealthProgressTool

try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"LangChain not available: {e}")

# Import medical tools
try:
    from tools import (
        # MedicalReadingsTool,
        SpecificMedicalValueTool,
        MultiPatientAnalysisTool,
        SimpleMedicalAnalysisTool,
        HospitalDocumentSearchTool,
        MedicationsTool,
        FoodlogTool,
        ProtocolTool,
        PlanTool,
        DoctorPatientMappingTool,
        UserProfileTool,
        DeviceTool,
        PatientSummaryTool
    )
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    print(f"Medical tools not available: {e}")

logger = logging.getLogger(__name__)

class MedicalLangChainAgent:
    """
    LangChain-based medical agent with conversation memory
    """
    
    def __init__(self, openai_api_key: str):
        """Initialize LangChain medical agent with tools and conversation tracking"""
        self.openai_api_key = openai_api_key
        self.agent_executor = None
        self.conversation_history = []  # Simple list to track conversations
        self.tools = []
        self.user_context = None  # Store user context for role-based access
        
        if LANGCHAIN_AVAILABLE and openai_api_key:
            self._setup_langchain_agent()
    
    def _generate_date_context(self) -> str:
        """Generate dynamic date context for the agent prompt"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        last_month = now.replace(day=1) - timedelta(days=1)
        
        return f"""**CURRENT DATE CONTEXT - CRITICAL:**
   - Today's date is {now.strftime('%B %d, %Y')}
   - "this month" → "{now.strftime('%Y-%m-01')}" ({now.strftime('%B %Y')})
   - "today" → "{now.strftime('%Y-%m-%d')}"
   - "yesterday" → "{yesterday.strftime('%Y-%m-%d')}"
   - "this week" → current week in {now.strftime('%B %Y')}
   - "last month" → "{last_month.strftime('%Y-%m-01')}" ({last_month.strftime('%B %Y')})
   - Always use {now.year} as the current year unless explicitly specified otherwise
   - For relative dates, calculate from {now.strftime('%B %d, %Y')}"""
    
    def _setup_langchain_agent(self):
        """Setup LangChain agent with medical tools and memory"""
        try:
            # Initialize OpenAI LLM
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=self.openai_api_key
            )
            
            # Create medical tools
            self.tools = self._create_medical_tools()

            print("TOOLS LOADED:", self.tools)
            
            # Generate current date context dynamically
            current_date_context = self._generate_date_context()
            
            # Create agent prompt with role-based instructions
            role_instructions = ""
            if self.user_context:
                if self.user_context.get('role_id') == 1:  # Patient role
                    role_instructions = f"""
🔒 **PATIENT ACCESS MODE**
- You are assisting a patient with their personal medical records
- All medical queries automatically show only your personal information
- Assume all queries are about YOUR medical data unless another patient is explicitly mentioned by name
- When the user asks "highest heart rate", "my glucose", "blood pressure on July 13th", "highest heart rate value on July 13th", etc. - these are YOUR personal medical queries
- Queries with general medical terms (without specific patient names) are about YOUR data
- Casual greetings and small talk (e.g., "hi", "hii", "hello", "hey", "good morning", "thanks") are NEVER patient name mentions — respond normally and ask how you can help
- Only restrict access when the message contains a clear first name and/or last name that matches or closely resembles a patient in the roster below (not any short word or greeting)
- If another patient is mentioned by name, respond: "I can only access your personal medical records. I cannot view other patients' information due to privacy and security restrictions."
"""
                else:  # Medical staff
                    role_instructions = f"""
👩‍⚕️ **MEDICAL STAFF ACCESS MODE**
- You are assisting: {self.user_context.get('role_name')} (User ID: {self.user_context.get('user_id')})
- You have access to all patient data as authorized medical personnel
- You can query specific patients by name or ID, or perform multi-patient analysis
- Always specify patient information when querying medical data

🔀 **MANDATORY TOOL ROUTING — glucose/sleep/activity/BP/TIR/HR questions:**
- If the question spans MORE THAN ONE DAY or asks for a trend/pattern/chart/summary
  over a period (words like "trend", "this week", "quality", "pattern", "over the
  last N days", or an explicit multi-day date range) — you MUST use
  get_health_progress. NEVER use get_specific_medical_value for these, and NEVER
  call get_specific_medical_value in a loop once per day to cover a range.
- Only use get_specific_medical_value when the question is about ONE exact
  moment or ONE single date (e.g. "glucose at 3pm", "sleep on July 5th").
- If you are unsure which case applies, use get_health_progress — it can also
  answer single-value questions.
"""
            
            # Patient database info - role-based visibility
            patient_db_info = ""
            if self.user_context and self.user_context.get('role_id') == 1:  # Patient role
                patient_db_info = """
🏥 **YOUR MEDICAL RECORDS:**
- All queries will automatically show your personal medical data
- Simply ask about any medical values (e.g., "highest heart rate on July 13th", "my glucose levels", "blood pressure this month")
- No need to specify your name - the system knows you're asking about your own data
- Questions like "What is the highest heart rate value on 13th July 2025" are about YOUR data
- Questions like "Show me glucose readings" are about YOUR data
- Only questions mentioning other patient names are restricted
"""
            else:  # Medical staff
                patient_db_info = """
🏥 **PATIENT ACCESS:**
Patients are identified by numeric patient_id. Any patient_id the user provides
(or that a tool resolves from a name) is valid — do not reject an ID on the
grounds that it's unfamiliar. If a tool call for a given patient_id returns
no data, say plainly that no data was found for that patient/date range —
never claim the patient doesn't exist or isn't "in the database" based on
your own memory of prior conversations.
"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a medical assistant AI for Revival Hospital. You help healthcare professionals and patients by analyzing medical data and answering questions about patient health records.

{role_instructions}

{patient_db_info}

🔧 **AVAILABLE TOOLS:**

1. **get_specific_medical_value** - Specific values with time/date filtering
   - Reading types: glucose, blood_pressure, spo2, body_temperature, hrv, stress, sleep, activity
   - Analysis types: highest (returns up to 10), lowest (returns up to 10), specific (returns 1)
   - Time ranges: morning, afternoon, evening, night
   - **For patients: NO need to specify patient_id or patient_name - automatically uses your data**
   - **For staff: Must specify patient_id or patient_name**
   - Use for: "List highest sugar levels in July", "Show 5 lowest BP readings", "highest heart rate on July 13th"
2. **analyze_multiple_patients** - Find DISTINCT patients with high/low values across all patients
   - Returns unique patients (not duplicate readings) with their highest/lowest values
   - Groups all readings per patient and shows summary with sample readings
   - Use for: "List all patients with high glucose", "Which patients had high BP on date X"
3. **get_medications** - Current medications and supplements (USE THIS FOR MEDICATION QUERIES)
   - Supports medication_type filter: "medication" or "supplement"
   - Returns top 10 latest records by default
   - Use for: "list medications", "current medications", "supplements", etc.
4. **get_foodlog** - Food log entries with type, description, and activity details
   - Returns top 10 latest food log records by default
   - Supports date filtering and patient search
   - Use for: "food intake", "food logs", "nutrition data", etc.
5. **get_protocols** - Treatment protocols and medical guidelines for patients
   - Returns detailed medical instructions, do's and don'ts, food protocols
   - Supports date filtering and patient search
   - **CRITICAL**: ALWAYS display the COMPLETE protocol content - never summarize or truncate
   - Use for: "treatment plans", "protocols", "medical guidelines", "care instructions", etc.
6. **get_my_plan** - Current plan details and usage summary
   - Returns plan information, benefits, consultation usage
   - Use for: "what's my plan", "plan details", "consultation usage", etc.
7. **get_doctor_patient_info** - Doctor-patient mapping and relationships
   - Query types: "my_doctor", "my_dha", "patient_primary_doctor", "doctor_patients"
   - Use for: "my doctor details", "who is my doctor", "doctor assignments", etc.
8. **get_user_profile** - Complete user profile with personal info and plans
   - Returns comprehensive profile: name, age, sex, contact info, active plans
   - Use for: "show my profile", "what's my age/sex", "profile details", etc.
9. **check_device_status** - Check device expiry status and count devices per patient
   - Checks if devices (especially CGM) are expired based on session_start_date + 15 days
   - Can check specific device or all devices for a patient
   - Supports patient name or patient ID
   - Use for: "When does my CGM expire?", "Is my CGM expired?", "How many devices does patient have?"
   - Parameters: patient_identifier (name or ID), device_name (default "CGM"), check_all_devices (boolean)
10. **search_hospital_documents** - Search hospital documents, policies, and medical protocols
11. **get_patient_summary** - Generate patient health summary

   Parameters:
   - query_type: "overall" OR "daily"
   - date: optional (YYYY-MM-DD)

   Rules:
   - "give me summary" → query_type="overall"
   - "today summary" → query_type="daily", date=today
   - "yesterday summary" → query_type="daily", date=yesterday
   - "summary for March 26 2026" → query_type="daily", date="2026-03-26"

   ALWAYS pass correct parameters.
📊 **MEDICAL DATA TYPES:**
- Glucose readings (Sugar levels)
- Blood pressure readings (BP with systolic/diastolic)
- Body temperature readings
- Sleep readings details (duration in hours/minutes from deep and light sleep only)
- HRV readings (Heart Rate Variability)
- SpO2 readings (Blood oxygen saturation)
- Stress readings
- Activity data (exercise, calories, steps, distance)
- Medications & Supplements
- Food intake & nutrition logs
- Treatment protocols & guidelines
- Patient treatment plans (plan details and descriptions)
- Medical devices (CGM devices with expiry tracking)

🎯 **QUERY HANDLING STRATEGY:**

**FOR PATIENT ROLE - CRITICAL UNDERSTANDING:**
- ALL medical queries without explicit patient names are about the current patient's data
- "What is the highest heart rate value on 13th July 2025" = patient's own data ✅
- "Show glucose readings this month" = patient's own data ✅
- "My blood pressure yesterday" = patient's own data ✅
- "Rayudu's glucose levels" = other patient's data ❌ (privacy restriction)
- When in doubt, assume it's about the patient's own data unless another name is mentioned

**PRIMARY TOOL SELECTION (Check first):**
- Patient-specific medical values → get_specific_medical_value
- **Multi-patient analysis** → analyze_multiple_patients (returns DISTINCT patients, not duplicate readings)
  - "List patients with high glucose" → analyze_multiple_patients
  - "Which patients had high BP on July 16th" → analyze_multiple_patients with date_filter="2025-07-16"
  - "Patients with high SpO2 this month" → analyze_multiple_patients with date_filter="{datetime.now().strftime('%Y-%m-01')}"
  - "Show patients with low values today" → analyze_multiple_patients with date_filter="{datetime.now().strftime('%Y-%m-%d')}"
  - Always returns unique patients with grouped readings and summaries
  - For relative dates like "this month", "today", "yesterday" - convert to proper YYYY-MM-DD format using current date context
- **MEDICATION QUERIES** → get_medications (with medication_type filter)
  - "list medications" → get_medications with medication_type="medication"
  - "list supplements" → get_medications with medication_type="supplement"
  - "current medications" → get_medications with medication_type="medication"
  - "latest medications" → get_medications with medication_type="medication"
- **PLAN QUERIES** → get_my_plan
  - "my plan", "my treatment plan", "show my plan" → get_my_plan
  - "plan for [patient_name]" → get_my_plan with patient_name
  - "what is my current plan" → get_my_plan
  - Always returns plan details with master plan information
- **PROFILE QUERIES** → get_user_profile
  - "show my profile", "what's my age/sex", "my details" → get_user_profile
  - "profile for patient X" → get_user_profile with patient_id
  - "show profile with plans" → get_user_profile (includes plans by default)
- **DOCTOR QUERIES** → get_doctor_patient_info
  - "my doctor", "who is my doctor", "doctor details" → query_type="my_doctor"
  - "my DHA details" → query_type="my_dha"
  - "patients for doctor X" → query_type="doctor_patients"
- Food intake/nutrition → get_foodlog
- Treatment protocols/guidelines → get_protocols
- **Sleep data** → get_specific_medical_value with reading_type="sleep"
  - "What is my sleep hours today" → get_specific_medical_value with reading_type="sleep", date_filter="2025-08-11"
  - "How many hours did I sleep yesterday" → get_specific_medical_value with reading_type="sleep", date_filter="2025-08-10"
  - "Sleep duration on August 6th" → get_specific_medical_value with reading_type="sleep", date_filter="2025-08-06"
  - Note: Sleep duration only includes deep sleep and light sleep, excluding REM and awake time

**FALLBACK TOOL SELECTION:**
- **IF NO OTHER TOOL MATCHES** → ALWAYS use search_hospital_documents
- **For general medical questions** → use search_hospital_documents
- **For medical terminology/definitions** → use search_hospital_documents
- **For hospital policies/procedures** → use search_hospital_documents
- **For medical protocols** → use search_hospital_documents
- **For unknown medical terms** → use search_hospital_documents

🔍 **CRITICAL INSTRUCTIONS:**

1. **MEDICATION QUERIES - SPECIAL HANDLING**:
   - For "list medications", "current medications", "latest medications" → ALWAYS use get_medications
   - For "list supplements", "current supplements", "latest supplements" → ALWAYS use get_medications
   - NEVER use get_specific_medical_value for medication/supplement queries
   - ALWAYS specify medication_type parameter: "medication" or "supplement"

2. **PLAN QUERIES - SPECIAL HANDLING**:
   - For "my plan", "what's my plan", "show my plan", "current plan" → ALWAYS use get_my_plan
   - For "plan details", "treatment plan", "plan benefits" → ALWAYS use get_my_plan
   - For "plan usage", "consultations left", "plan summary" → ALWAYS use get_my_plan with plan_type="summary"
   - NEVER use search_hospital_documents for patient-specific plan queries

3. **PROFILE QUERIES - SPECIAL HANDLING**:
   - For "show my profile", "what's my age", "my details", "my info" → ALWAYS use get_user_profile
   - For "profile with plans", "show my profile and plan" → ALWAYS use get_user_profile
   - For staff: "profile for patient X" → use get_user_profile with patient_id
   - NEVER use search_hospital_documents for patient-specific profile queries

4. **DOCTOR QUERIES - SPECIAL HANDLING**:
   - For "my doctor", "who is my doctor", "doctor details" → ALWAYS use get_doctor_patient_info with query_type="my_doctor" (patient role)
   - For "my DHA details", "DHA information" → ALWAYS use get_doctor_patient_info with query_type="my_dha" (patient role)
   - For staff: "list my patients", "my patients", "who are my patients" → use get_doctor_patient_info with query_type="my_patients" (no doctor_id needed — uses the logged-in staff member automatically)
   - For staff: "patients for doctor X" / "patients assigned to doctor 1212" → use get_doctor_patient_info with query_type="doctor_patients", doctor_id or doctor_name
   - NEVER use search_hospital_documents for doctor-patient relationship queries

5. **DEVICE QUERIES - SPECIAL HANDLING**:
   - For "When does my CGM expire?", "Is my CGM expired?" → ALWAYS use check_device_status
   - For "How many devices does patient have?" → use check_device_status with check_all_devices=true
   - For "Show all devices for [patient]" → use check_device_status with check_all_devices=true
   - For "Check [device] status for [patient]" → use check_device_status with specific device_name
   - Parameters: patient_identifier (name or ID), device_name (default "CGM"), check_all_devices (boolean)
   - Supports both patient names and patient IDs for identification
   - Returns expiry status (expired/not expired) and device counts
   - NEVER use search_hospital_documents for device expiry queries

6. **Tool Priority Logic**:
   - FIRST: Check if query matches patient-specific data tools
   - Medical definitions (like "MTP", "ICU protocols", etc.) → search_hospital_documents
   - General medical questions → search_hospital_documents
   - Hospital procedures → search_hospital_documents
   - Medical terminology → search_hospital_documents
   - Unknown medical abbreviations → search_hospital_documents
7. SUMMARY QUERIES - SPECIAL HANDLING (HIGHEST PRIORITY):

For ANY summary request, ALWAYS call get_patient_summary with parameters:

    - "give me summary"  
    → query_type="overall"

    - "today summary"  
    → query_type="daily", date=today

    - "yesterday summary"  
    → query_type="daily", date=yesterday

    - "summary for <date>"  
    → query_type="daily", date="<YYYY-MM-DD>"

    🚨 CRITICAL:
    - NEVER call tool without parameters
    - ALWAYS include query_type
    - INCLUDE date when applicable

3. **Patient Identification**: Always identify patients by name or ID. Use exact names from the database.

4. **Time & Date Parsing**: 
   - Parse natural language dates/times into proper formats
   - "16th July 2025" → "2025-07-16" (specific date)
   - "10 AM 16th July 2025" → "2025-07-16 10:00:00" (specific time)
   - "night time" → time_range="night"
   - **MONTH QUERIES - CRITICAL:**
     * "month of July" → date_filter="2025-07" (MONTH FORMAT, NOT DAY)
     * "July 2025" → date_filter="2025-07" (MONTH FORMAT)
     * "this month" → date_filter="{datetime.now().strftime('%Y-%m')}" (MONTH FORMAT)
     * "last month" → date_filter="{(datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')}" (MONTH FORMAT)
   - Use YYYY-MM format for month queries, YYYY-MM-DD format for specific dates only
   
   {current_date_context}

5. **Value Interpretation**: 
   - Glucose: Normal 70-140 mg/dL, High >180, Low <70
   - Blood Pressure: Normal <120/80, High >140/90
   - Handle pronouns (he/she/they) referring to last mentioned patient
   - Maintain conversation context for follow-up questions

**EXAMPLE SCENARIOS:**
- "List current supplements for Rayudu" → use get_medications with medication_type="supplement"
- "Show treatment protocols for Eswar" → use get_protocols (SHOW COMPLETE CONTENT)
- "What are the dietary guidelines for patient 111?" → use get_protocols (SHOW ALL GUIDELINES)
- "Get care instructions for Rayudu" → use get_protocols (DISPLAY FULL INSTRUCTIONS)
- "Show food protocols for Eswar" → use get_protocols (COMPLETE FOOD PROTOCOL)
- "Latest protocol for Rayudu" → use get_protocols (FULL PROTOCOL DETAILS)
- "When does my CGM expire?" → use check_device_status (patient asking about their own device)
- "Is patient 132's CGM expired?" → use check_device_status with patient_identifier="132"
- "How many devices does Rayudu have?" → use check_device_status with patient_identifier="Rayudu", check_all_devices=true
- "Show all devices for patient 111" → use check_device_status with patient_identifier="111", check_all_devices=true
- "Check blood pressure monitor for Eswar" → use check_device_status with patient_identifier="Eswar", device_name="Blood Pressure Monitor"
- "List patients with high SpO2 this month" → use analyze_multiple_patients with date_filter="{datetime.now().strftime('%Y-%m-01')}"
- "Patients with high glucose today" → use analyze_multiple_patients with date_filter="{datetime.now().strftime('%Y-%m-%d')}"
- "Show patients with low BP yesterday" → use analyze_multiple_patients with date_filter="{(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}"
- "Infection control protocols?" → use search_hospital_documents (hospital procedure)
- "Emergency procedures?" → use search_hospital_documents (hospital procedure)

**PATIENT-SPECIFIC EXAMPLES (for patient role):**
- "What is the highest heart rate value on 13th July 2025" → use get_specific_medical_value with reading_type="hrv", date_filter="2025-07-13", analysis_type="highest"
- "my glucose levels this month" → use get_specific_medical_value with reading_type="glucose", date_filter="2025-08"
- "highest blood pressure yesterday" → use get_specific_medical_value with reading_type="blood_pressure", date_filter="{(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}", analysis_type="highest"
- "List the highest sugar levels in July" → use get_specific_medical_value with reading_type="glucose", date_filter="2025-07", analysis_type="highest"
- "lowest sugar level in July" → use get_specific_medical_value with reading_type="glucose", date_filter="2025-07", analysis_type="lowest"
- "When does my CGM expire?" → use check_device_status with patient_identifier=[current_user_id]
- "Is my CGM expired?" → use check_device_status with patient_identifier=[current_user_id], device_name="CGM"
- "How many devices do I have?" → use check_device_status with patient_identifier=[current_user_id], check_all_devices=true
- "What is my sleep hours today" → use get_specific_medical_value with reading_type="sleep", date_filter="2025-08-11"
- "How many hours did I sleep yesterday" → use get_specific_medical_value with reading_type="sleep", date_filter="2025-08-10"

🔍 **RESPONSE FORMATTING INSTRUCTIONS:**

1. **Protocol Data Display**:
   - When displaying protocol/treatment data, ALWAYS show the COMPLETE content from the description field
   - DO NOT summarize or truncate protocol information
   - Display the full protocol content exactly as stored in the database
   - If the protocol content is in HTML format, extract and display the readable text content
   - Show ALL sections including Do's, Don'ts, dietary guidelines, treatment instructions, etc.

2. **Complete Data Display**:
   - For protocol queries, user needs the FULL information for medical compliance
   - NEVER say "For more detailed information..." - provide ALL available details immediately
   - If data appears incomplete, explicitly state what might be missing
   - Present the data in a well-formatted, readable manner

3. **Sleep Data Responses**:
   - When reporting sleep duration, use the "total_sleep_duration" from the sleep data response
   - This includes deep, light, and REM sleep but excludes awake time
   - Example: "Today, you have slept for a total of 6 hours and 29 minutes." (includes all sleep stages except awake)
   - If breakdown is requested, use the sleep_breakdown data to provide details about deep sleep, light sleep, etc.

Remember: You provide data analysis and insights, not medical diagnosis. Always suggest consulting healthcare providers for concerning values or treatment decisions. For protocol and treatment queries, provide COMPLETE information as medical compliance requires full details."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])
            
            # Create agent
            agent = create_openai_tools_agent(llm, self.tools, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=5,  # Increased from 5 to 10 for better response completeness
                handle_parsing_errors=True,
                return_intermediate_steps=False,
                max_execution_time=120  # Increased to 120 seconds for complex protocol queries
            )
            
            logger.info("✅ Medical LangChain agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup medical LangChain agent: {e}")
            self.agent_executor = None
    
    def set_user_context(self, user_context: Dict[str, Any]):
        """Set user context for role-based access control"""
        self.user_context = user_context
        logger.info(f"User context set for agent: User {user_context.get('user_id')} (Role: {user_context.get('role_name')})")
        
        # Recreate tools with user context
        if LANGCHAIN_AVAILABLE and self.openai_api_key:
            self.tools = self._create_medical_tools()
            # Recreate agent executor with updated tools
            self._setup_langchain_agent()

    def _create_medical_tools(self) -> List:
        """Create medical tools for LangChain agent with user context"""
        try:
            if not TOOLS_AVAILABLE:
                logger.warning("⚠️ Medical tools not available")
                return []
            
            # Create working medical tools and inject user context
            if self.user_context and self.user_context.get('role_id') == 1:  # Patient role
                patient_id = self.user_context.get('user_id')
                logger.info(f"Creating patient-restricted tools for patient ID: {patient_id}")
                
                # Create tools and set user context for role-based filtering
                tools = [
                    SpecificMedicalValueTool(),
                    SimpleMedicalAnalysisTool(),
                    MedicationsTool(),
                    FoodlogTool(),
                    ProtocolTool(),
                    PlanTool(),  # Allow patients to view their own plans
                    DoctorPatientMappingTool(),  # Allow patients to query their doctor details
                    UserProfileTool(),  # Allow patients to view their own profile
                    HospitalDocumentSearchTool(),  # Allow general hospital info
                    DeviceTool(),  # Allow patients to check their device expiry
                    PatientSummaryTool()
                ]
                
                # Set user context on each tool for role-based access
                for tool in tools:
                    try:
                        if hasattr(tool, 'set_user_context'):
                            tool.set_user_context(self.user_context)
                            logger.debug(f"✅ Set user context on {tool.__class__.__name__}")
                        else:
                            logger.debug(f"ℹ️ {tool.__class__.__name__} doesn't support user context")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to set user context on {tool.__class__.__name__}: {e}")
                
            else:
                # Medical staff - full access to all tools
                logger.info("Creating full-access tools for medical staff")
                tools = [
                    SpecificMedicalValueTool(),
                    MultiPatientAnalysisTool(),
                    SimpleMedicalAnalysisTool(),
                    HospitalDocumentSearchTool(),
                    MedicationsTool(),
                    FoodlogTool(),
                    ProtocolTool(),
                    PlanTool(),  # Staff can view any patient's plans
                    DoctorPatientMappingTool(),  # Staff can view all doctor-patient mappings
                    UserProfileTool(),  # Staff can view any patient's profile
                    DeviceTool() , # Staff can check any patient's device expiry
                    PatientSummaryTool(),
                    HealthProgressTool() # Doctor/DHA: glucose/BP/HR/stress/HRV/activity/sleep trends with chart data 
                ]
                
                # Set user context on each tool
                for tool in tools:
                    try:
                        if hasattr(tool, 'set_user_context'):
                            tool.set_user_context(self.user_context)
                            logger.debug(f"✅ Set user context on {tool.__class__.__name__}")
                        else:
                            logger.debug(f"ℹ️ {tool.__class__.__name__} doesn't support user context")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to set user context on {tool.__class__.__name__}: {e}")
            
            return tools
            
        except Exception as e:
            logger.error(f"❌ Failed to create medical tools: {e}")
            return []
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """
        Process a chat message with automatic tool selection and memory
        """
        try:
            if self.agent_executor and LANGCHAIN_AVAILABLE:
                # Add user message to history
                self.conversation_history.append({"role": "user", "content": message})
                
                # Truncate conversation history to manage tokens
                truncated_history = self.truncate_conversation_history(
                    self.conversation_history[:-1],  # Exclude current message
                    12000,  # max tokens
                    20      # max messages
                )
                
                # Convert truncated history to LangChain format
                chat_history = []
                for msg in truncated_history:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
                
                # Use LangChain agent executor with managed chat history
                logger.debug(f"🎯 LangChain processing: {message[:100]}...")
                logger.debug(f"📚 Chat history messages: {len(chat_history)}")
                
                response = await self.agent_executor.ainvoke({
                    "input": message,
                    "chat_history": chat_history
                })
                
                # Add AI response to history
                if response.get("output"):
                    self.conversation_history.append({"role": "assistant", "content": response["output"]})
                
                                # Pick up any chart data a tool stashed on itself during this run
                # (a LangChain tool can only return text to the LLM, so tools that
                # produce a chart use this side-channel instead). Cleared after
                # reading so a stale chart never leaks into a later unrelated answer.
                chart_data = None
                for tool in self.tools:
                    pending = getattr(tool, 'last_chart_data', None)
                    if pending is not None:
                        chart_data = pending
                        object.__setattr__(tool, 'last_chart_data', None)
                        break  # one chart per response — first one found wins

                return {
                    "message": response["output"],
                    "chart_data": chart_data,
                    "metadata": {
                        "agent_type": "Revival365AI Agent",
                        "memory_messages": len(self.conversation_history),
                        "timestamp": datetime.now().isoformat(),
                        "tools_available": len(self.tools),
                        "response_length": len(response.get("output", ""))
                    }
                }
            else:
                # No agent available
                return {
                    "message": "Medical agent not available to process the request.",
                    "metadata": {"error": True}
                }
                    
        except Exception as e:
            logger.error(f"Medical chat processing failed: {e}")
            return {
                "message": f"Sorry, I encountered an error: {str(e)}",
                "metadata": {"error": True}
            }
    
    def truncate_conversation_history(self, conversation_history: List[Dict[str, Any]], 
                                    max_tokens: int = 12000,
                                    max_messages: int = 20) -> List[Dict[str, Any]]:
        """Truncate conversation history to stay within token limits"""
        if not conversation_history:
            return []
        
        # First, limit by number of messages
        if len(conversation_history) > max_messages:
            conversation_history = conversation_history[-max_messages:]
        
        # Simple token estimation (4 chars per token)
        total_chars = 0
        truncated_history = []
        
        # Start from most recent and work backwards
        for msg in reversed(conversation_history):
            msg_chars = len(msg.get("content", ""))
            if total_chars + msg_chars > max_tokens * 4:  # rough estimation
                break
            total_chars += msg_chars
            truncated_history.insert(0, msg)
        
        return truncated_history
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
