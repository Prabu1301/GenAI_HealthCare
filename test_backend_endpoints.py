import requests

# 📄 Dummy transcript
transcript_text = """
Patient Metadata: Name: John Doe | Age: 34 | ID: ENT1025 | Visit Type: Follow-up
Summary: Patient presents with ear pain and mild hearing loss for the past 3 days. Examination shows mild otitis externa without discharge.
Advised antibiotic ear drops and review after one week.
"""

# 🧑‍⚕️ Patient metadata
patient_id = "ENT1025"
metadata = {
    "name": "John Doe",
    "age": 34,
    "visit_type": "Follow-up"
}

# 🧠 Structured Notes (as dictionary)
structured_notes = {
    "chief_complaint": "Ear pain and mild hearing loss",
    "assessment": "Mild otitis externa without discharge",
    "plan": "Antibiotic ear drops and review in one week"
}

# 🧠 CDS Insights (as dictionary)
cds_insights = {
    "diagnosis": "Otitis externa",
    "recommendation": "Start topical antibiotics"
}

# 🔐 Optional: Add auth headers if required
headers = {
    # "Authorization": "Bearer YOUR_TOKEN",  # Uncomment and insert token if needed
    # "X-Api-Key": "YOUR_API_KEY"
}

# ✅ 1. Structure Notes
response1 = requests.post(
    "http://127.0.0.1:8000/structure-notes",
    json={"transcript": transcript_text},
    headers=headers
)
print("🧾 /structure-notes:", response1.status_code, response1.json())

# ✅ 2. Generate CDS Insights
response2 = requests.post(
    "http://127.0.0.1:8000/cds-insights",
    json={"structured_notes": structured_notes},
    headers=headers
)
print("🧠 /cds-insights:", response2.status_code, response2.json())

# ✅ 3. Generate Report
response3 = requests.post(
    "http://127.0.0.1:8000/generate-report",
    json={
        "patient_id": patient_id,
        "transcript": transcript_text,
        "structured_notes": structured_notes,
        "cds_insights": cds_insights,
        "patient_metadata": metadata
    },
    headers=headers
)
print("📄 /generate-report:", response3.status_code, response3.json())

# ✅ 4. Update Report
response4 = requests.put(
    f"http://127.0.0.1:8000/update-report/{patient_id}",
    json={"corrections": transcript_text},
    headers=headers
)
print("✏️ /update-report:", response4.status_code, response4.json())

# ✅ 5. Get Patient Reports
response5 = requests.get(
    f"http://127.0.0.1:8000/reports/{patient_id}",
    headers=headers
)
print("📁 /reports:", response5.status_code, response5.json())