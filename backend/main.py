from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import json
import anthropic
import whisper
from datetime import datetime
import uuid
from pathlib import Path

app = FastAPI(title="GenAI Healthcare Documentation API")

# Configuration
BASE_DIR = Path("C:/GitHub/GenAI_HealthCare")
AUDIO_DIR = BASE_DIR / "audio_files"
REPORTS_DIR = BASE_DIR / "reports"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"

# Ensure audio_files directory exists
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    audio_path = AUDIO_DIR / file.filename
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "saved_to": str(audio_path)}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        audio_path = AUDIO_DIR / file.filename
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))

        return {"filename": file.filename, "transcript": result["text"]}
    except Exception as e:
        print(f"Transcription error: {e}")
        return {"error": str(e)}

# Create directories
for dir_path in [AUDIO_DIR, REPORTS_DIR, TRANSCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize clients
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
whisper_model = whisper.load_model("base")

# Pydantic Models
class TranscriptionRequest(BaseModel):
    audio_path: str

class StructureNotesRequest(BaseModel):
    transcript: str
    patient_id: Optional[str] = None

class CDSRequest(BaseModel):
    structured_notes: Dict
    patient_history: Optional[Dict] = None

class ReportRequest(BaseModel):
    patient_id: str
    transcript: str
    structured_notes: Dict
    cds_insights: Dict

class UpdateReportRequest(BaseModel):
    corrections: str
    regenerate_cds: bool = False

# Prompt Templates
SOAP_PROMPT = """You are a medical documentation assistant. Convert the following doctor's dictation into a structured SOAP note format.

Dictation: {transcript}

Create a structured SOAP note with these sections:
- **Subjective**: Patient's complaints, symptoms, history
- **Objective**: Physical examination findings, vital signs, lab results
- **Assessment**: Diagnosis or clinical impressions
- **Plan**: Treatment plan, medications, follow-up

Return as JSON with this structure:
{{
  "subjective": {{
    "chief_complaint": "",
    "history_present_illness": "",
    "symptoms": []
  }},
  "objective": {{
    "vital_signs": {{}},
    "physical_exam": {{}},
    "lab_results": {{}}
  }},
  "assessment": {{
    "diagnoses": [],
    "clinical_impression": ""
  }},
  "plan": {{
    "medications": [],
    "procedures": [],
    "follow_up": "",
    "patient_education": []
  }}
}}

Ensure all medical terminology is accurate and properly formatted."""

CDS_PROMPT = """You are a Clinical Decision Support system. Analyze the following structured clinical notes and provide evidence-based insights.

Structured Notes: {structured_notes}

Patient History: {patient_history}

Provide CDS insights in this JSON format:
{{
  "diagnosis_suggestions": [
    {{
      "condition": "",
      "confidence": "high/medium/low",
      "supporting_evidence": [],
      "icd10_code": ""
    }}
  ],
  "alerts": [
    {{
      "type": "drug_interaction/contraindication/allergy",
      "severity": "critical/high/medium/low",
      "message": "",
      "recommendation": ""
    }}
  ],
  "treatment_recommendations": [
    {{
      "intervention": "",
      "evidence_level": "A/B/C",
      "rationale": "",
      "alternatives": []
    }}
  ],
  "preventive_care": [
    {{
      "screening": "",
      "due_date": "",
      "priority": "high/medium/low"
    }}
  ],
  "red_flags": []
}}

Focus on patient safety, evidence-based medicine, and actionable recommendations."""

SUMMARY_PROMPT = """Generate a comprehensive clinical summary report from the following data:

Transcript: {transcript}

Structured Notes: {structured_notes}

CDS Insights: {cds_insights}

Create a professional medical summary that includes:
1. Executive Summary (2-3 sentences)
2. Key Findings
3. Diagnoses with ICD-10 codes
4. Treatment Plan
5. Critical Alerts
6. Follow-up Actions
7. Clinical Recommendations

Format as a clear, concise report suitable for medical records."""

# Helper Functions
def get_next_version(patient_id: str) -> int:
    """Get the next version number for a patient's report."""
    pattern = f"{patient_id}_v*.json"
    existing = list(REPORTS_DIR.glob(pattern))
    if not existing:
        return 1
    versions = [int(f.stem.split('_v')[1]) for f in existing]
    return max(versions) + 1

def save_report(patient_id: str, data: Dict) -> str:
    """Save report with versioning."""
    version = get_next_version(patient_id)
    filename = f"{patient_id}_v{version}.json"
    filepath = REPORTS_DIR / filename
    
    data['metadata'] = {
        'patient_id': patient_id,
        'version': version,
        'created_at': datetime.now().isoformat(),
        'report_id': str(uuid.uuid4())
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return str(filepath)

def call_claude(prompt: str, system: str = "You are a medical AI assistant.") -> str:
    """Call Claude API with structured prompts."""
    message = claude_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# API Endpoints
@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for processing."""
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        raise HTTPException(400, "Unsupported audio format")
    
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    filename = f"{file_id}.{ext}"
    filepath = AUDIO_DIR / filename
    
    content = await file.read()
    with open(filepath, 'wb') as f:
        f.write(content)
    
    return {
        "status": "success",
        "file_id": file_id,
        "filepath": str(filepath),
        "filename": filename
    }

@app.post("/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    """Transcribe audio using Whisper."""
    audio_path = request.audio_path
    
    if not os.path.exists(audio_path):
        raise HTTPException(404, "Audio file not found")
    
    try:
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]
        
        # Save transcript
        file_id = Path(audio_path).stem
        transcript_path = TRANSCRIPTS_DIR / f"{file_id}.txt"
        with open(transcript_path, 'w') as f:
            f.write(transcript)
        
        return {
            "status": "success",
            "transcript": transcript,
            "transcript_path": str(transcript_path),
            "language": result.get("language"),
            "duration": len(result.get("segments", []))
        }
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

@app.post("/structure-notes")
async def structure_notes(request: StructureNotesRequest):
    """Convert transcript to structured SOAP notes using Claude."""
    try:
        prompt = SOAP_PROMPT.format(transcript=request.transcript)
        response = call_claude(prompt, "You are an expert medical documentation specialist.")
        
        # Parse JSON from response
        structured_notes = json.loads(response)
        
        return {
            "status": "success",
            "structured_notes": structured_notes
        }
    except json.JSONDecodeError:
        raise HTTPException(500, "Failed to parse structured notes")
    except Exception as e:
        raise HTTPException(500, f"Structure notes failed: {str(e)}")

@app.post("/cds-insights")
async def generate_cds_insights(request: CDSRequest):
    """Generate Clinical Decision Support insights."""
    try:
        prompt = CDS_PROMPT.format(
            structured_notes=json.dumps(request.structured_notes, indent=2),
            patient_history=json.dumps(request.patient_history or {}, indent=2)
        )
        response = call_claude(prompt, "You are a Clinical Decision Support system providing evidence-based medical insights.")
        
        cds_insights = json.loads(response)
        
        return {
            "status": "success",
            "cds_insights": cds_insights
        }
    except json.JSONDecodeError:
        raise HTTPException(500, "Failed to parse CDS insights")
    except Exception as e:
        raise HTTPException(500, f"CDS generation failed: {str(e)}")

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate comprehensive clinical summary report."""
    try:
        prompt = SUMMARY_PROMPT.format(
            transcript=request.transcript,
            structured_notes=json.dumps(request.structured_notes, indent=2),
            cds_insights=json.dumps(request.cds_insights, indent=2)
        )
        summary = call_claude(prompt, "You are a medical documentation specialist creating clinical summaries.")
        
        # Bundle all data
        report_data = {
            "patient_id": request.patient_id,
            "transcript": request.transcript,
            "structured_notes": request.structured_notes,
            "cds_insights": request.cds_insights,
            "clinical_summary": summary,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save with versioning
        filepath = save_report(request.patient_id, report_data)
        
        return {
            "status": "success",
            "report_path": filepath,
            "report_data": report_data
        }
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {str(e)}")

@app.put("/update-report/{patient_id}")
async def update_report(patient_id: str, request: UpdateReportRequest):
    """Update existing report with corrections."""
    # Get latest version
    pattern = f"{patient_id}_v*.json"
    existing = list(REPORTS_DIR.glob(pattern))
    
    if not existing:
        raise HTTPException(404, "No existing report found")
    
    latest = max(existing, key=lambda x: int(x.stem.split('_v')[1]))
    
    with open(latest, 'r') as f:
        original_data = json.load(f)
    
    # Apply corrections using Claude
    correction_prompt = f"""Original Clinical Summary:
{original_data['clinical_summary']}

Corrections to apply:
{request.corrections}

Generate an updated clinical summary incorporating these corrections while maintaining medical accuracy and proper documentation standards."""
    
    updated_summary = call_claude(correction_prompt)
    
    # Create updated report
    updated_data = original_data.copy()
    updated_data['clinical_summary'] = updated_summary
    updated_data['corrections_applied'] = request.corrections
    updated_data['updated_at'] = datetime.now().isoformat()
    
    # Regenerate CDS if requested
    if request.regenerate_cds:
        cds_prompt = CDS_PROMPT.format(
            structured_notes=json.dumps(original_data['structured_notes'], indent=2),
            patient_history="{}"
        )
        updated_cds = call_claude(cds_prompt)
        updated_data['cds_insights'] = json.loads(updated_cds)
    
    # Save new version
    filepath = save_report(patient_id, updated_data)
    
    return {
        "status": "success",
        "report_path": filepath,
        "version": get_next_version(patient_id) - 1,
        "report_data": updated_data
    }

@app.get("/reports/{patient_id}")
async def get_patient_reports(patient_id: str):
    """Retrieve all versions of a patient's reports."""
    pattern = f"{patient_id}_v*.json"
    reports = list(REPORTS_DIR.glob(pattern))
    
    if not reports:
        raise HTTPException(404, "No reports found for this patient")
    
    report_list = []
    for report_path in sorted(reports):
        with open(report_path, 'r') as f:
            data = json.load(f)
            report_list.append({
                "version": data['metadata']['version'],
                "created_at": data['metadata']['created_at'],
                "report_id": data['metadata']['report_id'],
                "filepath": str(report_path)
            })
    
    return {
        "patient_id": patient_id,
        "total_versions": len(report_list),
        "reports": report_list
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "whisper_model": "loaded",
        "claude_api": "configured"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)