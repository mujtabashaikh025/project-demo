import streamlit as st
import google.generativeai as genai 
import pytesseract
from pdf2image import convert_from_bytes
import pandas as pd
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from dotenv import load_dotenv
import pypdf # NEW LIBRARY
import io

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
st.set_page_config(page_title="NAMA Compliance Agent", layout="wide")

# API Configuration
api_key = st.secrets["api_key"]
genai.configure(api_key=api_key)

REQUIRED_DOCS = [
    "1- Fees application receipt copy.",
    "2- Nama water services vendor registeration certificates & Product Agency certificates or authorization letter from Factory for local distributor ratified from Oman embassy.",
    "3- Certificate of incorporation of the firm (Factory & Foundry).",
    "4- Manufacturing Process flow chart of product and list of out sourced process / operation if applicable including Outsourcing name & address.",
    "5-Valid copies certificates of (ISO 9001, ISO 45001 & ISO 14001).",
    "6- Factory Layout chart.",
    "7-Factory Organizational structure, Hierarchy levels, Ownership details.",
    "8- Product Compliance Statement with reference to Nama water services specifications (with supports documents accordingly).",
    "9- Product Technical datasheets.",
    "10- Omanisation details from Ministry of Labour.",
    "11- Product Independent Test certificates.",
    "12- Attestation of Sanitary Conformity (hygiene test including mechanical assessment for a full product certificate at 50 degrees Celsiusfull to used in drinking water)",
    "13- Provide products Chemicals Composition of materials.",
    "14- Reference list of products used in Oman or any GCC projects with contact no. or emails of end user or clients."
]

# --- 2. INTELLIGENT EXTRACTION (Hybrid: Text First -> OCR Fallback) ---
def extract_text_smart(uploaded_file):
    """
    Attempts to read text directly. 
    If text < 50 chars (likely scanned), falls back to OCR.
    """
    text = ""
    file_bytes = uploaded_file.getvalue()
    
    try:
        # METHOD 1: Direct Text Extraction (Super Fast)
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        # Limit to first 3 pages
        num_pages = len(pdf_reader.pages)
        limit = min(3, num_pages)
        
        for i in range(limit):
            page_text = pdf_reader.pages[i].extract_text()
            if page_text:
                text += page_text

        # If we found substantial text, return it immediately
        if len(text.strip()) > 100: 
            return f"FILE_NAME: {uploaded_file.name}\n(Extracted via Text Layer)\n{text[:15000]}"

    except Exception as e:
        print(f"Direct extract failed for {uploaded_file.name}: {e}")

    # METHOD 2: Fallback to OCR (Slower, but necessary for scans)
    try:
        # Convert to images with LOWER DPI (150 is usually enough for text) to speed up
        images = convert_from_bytes(file_bytes, first_page=1, last_page=3, dpi=150, grayscale=True)
        
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
            
        return f"FILE_NAME: {uploaded_file.name}\n(Extracted via OCR)\n{ocr_text[:15000]}"
    except Exception as e:
        return f"Error reading {uploaded_file.name}: {str(e)}"

def batch_extract_all(files):
    """Uses ThreadPoolExecutor to process files simultaneously."""
    # Increased workers since direct extraction is not CPU bound
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(extract_text_smart, files))
    return results

# --- 3. BATCHED AI ANALYSIS ---
def analyze_batch(batch_text_list):
    model = genai.GenerativeModel('gemini-2.5-pro') # Use Flash for speed
    today_str = date.today().strftime("%Y-%m-%d")

    prompt = f"""
    Today is {today_str}. You are NAMA Document Analyzer.
    Extract data from pdfs and translate it if it is not in english.
    Classify each document using this list: {json.dumps(REQUIRED_DOCS)}
    
    Compliance Rule: ISO certificates must be valid for >180 days from {today_str}.
    
    Return ONLY a JSON object with this EXACT structure:
    {{
        "iso_analysis": [
            {{
                "standard": "ISO 9001",
                "expiry_date": "YYYY-MM-DD",
                "days_remaining": 0,
                "compliance_status": "Pass/Fail"
            }}
        ],
        "found_documents": [
            {{"filename": "name.pdf", "Category": "Category from list", "Status": "Valid"}}
        ],
        "wras_analysis": {{
                "found": true,
                "wras_id": "123456"
            }}
    }}
    """
    
    combined_content = "\n\n=== NEXT DOCUMENT ===\n".join(batch_text_list)
    
    try:
        response = model.generate_content(
            contents=[prompt, combined_content],
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        if isinstance(data, list): return data[0]
        return data
    except Exception:
        return {}

# --- 4. ONLINE WRAS CHECKER ---
def verify_wras_online(wras_id):
    if not wras_id or wras_id == "N/A":
        return {"status": "Skipped", "url": "#"}

    search_url = f"https://www.wrasapprovals.co.uk/approvals-directory/?search={wras_id}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers, timeout=5)
        if response.status_code == 200 and "No results found" not in response.text:
            return {"status": "Active", "online_id": wras_id, "url": search_url}
        return {"status": "Not Found", "url": search_url}
    except:
        return {"status": "Error", "url": search_url}

# --- 5. UI & EXECUTION LOGIC ---
st.title("NAMA Compliance AI Audit")

uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

if st.button("Run Audit", type="primary"):
    if uploaded_files:
        start_time = datetime.now()
        
        # 1. Fast Extraction
        with st.status("Reading Documents...", expanded=True) as status:
            st.write("Extracting text...")
            all_texts = batch_extract_all(uploaded_files)
            status.update(label="Text Extraction Complete!", state="complete", expanded=False)

        # 2. Parallel AI Analysis
        final_report = {
            "iso_analysis": [],
            "wras_analysis": {"found": False, "wras_id": None},
            "found_documents": [],
            "missing_documents": set(REQUIRED_DOCS),
            "wras_online_check": {"status": "N/A", "url": "#"}
        }

        # Create batches
        batch_size = 8
        batches = [all_texts[i:i + batch_size] for i in range(0, len(all_texts), batch_size)]
        
        with st.spinner(f"Analyzing {len(batches)} batches with AI..."):
            # Execute AI batches in PARALLEL
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_batch = {executor.submit(analyze_batch, batch): batch for batch in batches}
                
                for future in as_completed(future_to_batch):
                    batch_res = future.result()
                    if isinstance(batch_res, dict):
                        final_report["iso_analysis"].extend(batch_res.get("iso_analysis", []))
                        final_report["found_documents"].extend(batch_res.get("found_documents", []))
                        
                        wras = batch_res.get("wras_analysis", {})
                        if isinstance(wras, dict) and wras.get("found"):
                            final_report["wras_analysis"] = wras

        # Post-Processing
        for doc in final_report["found_documents"]:
            doc_type = doc.get("Category")
            if doc_type in final_report["missing_documents"]:
                final_report["missing_documents"].remove(doc_type)

        extracted_id = final_report["wras_analysis"].get("wras_id")
        if extracted_id:
            final_report["wras_online_check"] = verify_wras_online(extracted_id)

        st.session_state.analysis_result = final_report
        
        duration = (datetime.now() - start_time).total_seconds()
        st.success(f"Audit Complete in {duration:.2f} seconds!")

# --- 6. DISPLAY RESULTS (Same as before) ---
if "analysis_result" in st.session_state:
    res = st.session_state.analysis_result
    no_of_missing_docs = len(res["missing_documents"])
    doc_score = round(((14 - no_of_missing_docs) / 14) * 100, 2)
    
    wras_data = res.get("wras_online_check", {})
    wras_url = wras_data.get("url", "#")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💧 WRAS Status", wras_data.get("status", "N/A"), border=True)
        if wras_url != "#": st.link_button("🔍 Verify", wras_url)
    
    col2.metric("⛔ Missing Docs", f"{no_of_missing_docs}", border=True)
    col3.metric("🏆 Score", f"{doc_score}%", border=True)

    st.subheader("❌ Missing Documents")
    if res["missing_documents"]:
        for m in sorted(list(res["missing_documents"])):
            st.error(f"Missing: {m}")
    else:
        st.success("All required documents found!")
            
    st.subheader("✅ Documents Found")
    if res["found_documents"]:
        st.dataframe(pd.DataFrame(res["found_documents"]), use_container_width=True)

    st.subheader("🏭 ISO Validation")
    iso_data = res.get('iso_analysis', [])
    if iso_data:
        cols = st.columns(3)
        for idx, iso in enumerate(iso_data):
            with cols[idx % 3]:
                std_name = iso.get('standard', 'Unknown ISO')
                status = iso.get('compliance_status', 'Fail')
                days = iso.get('days_remaining', 0)
                
                status_color = "green" if "Pass" in status else "red"
                with st.container(border=True):
                    st.markdown(f"#### :{status_color}[{std_name}]")
                    if days < 180:
                        st.error(f"⚠️ {days} days left")
                    else:
                        st.success(f"✅ {days} days left")
                    st.caption(f"Expires: {iso.get('expiry_date')}")
