from cProfile import label
import re
IOLATION_KEYWORDS = [
    "ละเมิด", "ปลอมแปลง", "เลียนแบบ", "ทำซ้ำ", "ดัดแปลง",
    "เผยแพร่", "จำหน่าย", "ละเมิดสิทธิ", "ละเมิดลิขสิทธิ์",
    "ละเมิดสิทธิบัตร", "การละเมิด", "กระทำผิด", "ฝ่าฝืน",
]

PATENT_KEYWORDS = [
    "สิทธิบัตร", "การประดิษฐ์", "เครื่องหมายการค้า",
    "สิทธิบัตรการประดิษฐ์", "แบบอรรถประโยชน์", "อนุสิทธิบัตร",
    "จดทะเบียนสิทธิบัตร", "คำขอรับสิทธิบัตร", "ผู้ทรงสิทธิ",
]

COPYRIGHT_KEYWORDS = [
    "ลิขสิทธิ์", "วรรณกรรม", "ดนตรีกรรม", "ภาพยนตร์",
    "สิ่งบันทึกเสียง", "งานสร้างสรรค์", "ผู้สร้างสรรค์",
    "เจ้าของลิขสิทธิ์", "ซอฟต์แวร์", "โปรแกรมคอมพิวเตอร์",
]
TRADEMARK_KEYWORDS = [
    "เครื่องหมายการค้า",
    "ไม่ได้รับอนุญาต",
    "ปลอมแปลง",
]
def detect_category(text):
    is_patent = any(re.search(k,text)for k in PATENT_KEYWORDS)
    is_copyright = any(re.search(k,text)for k in COPYRIGHT_KEYWORDS)
    is_trademark = any(re.search(k,text)for k in TRADEMARK_KEYWORDS)
    is_iolation = any(re.search(k,text)for k in IOLATION_KEYWORDS)
    
    if is_patent: return "patent"
    if is_copyright: return "copyright"
    if is_trademark: return "trademark" 
    if is_iolation: return "iolation"
    return 0
sample = "การละเมิดสิทธิบัตรเป็นปัญหาที่สำคัญในอุตสาหกรรมเทคโนโลยี"
print(f"text: {sample}")
print(f"Predicted class: {detect_category(sample)}")

def cal_confidence(text, predicted_class):
    base_confidence = 0.7
    signal = []
    if "มาตรา" in text or "พ.ร.บ." in text:
        base_confidence = 0.15
        signal.append("statotury_reference")
        
    if "คำพิพากษา" in text or "ศาล" in text:
        base_confidence = 0.1
        signal.append("precedent_reference")
    return  min(base_confidence, 0.8), signal    
        
text_with_statute = "ละเมิดสิทธิบัตรตามมาตรา 123 ของพ.ร.บ.สิทธิบัตร"
conf, sig = cal_confidence(text_with_statute, "patent")
print(f"text: {text_with_statute}")
print(f"Confidence: {conf}, Signals: {sig}")   
print(f"Predicted class: patent, Confidence: {conf}, Signals: {sig}")

def get_physic_gate_preview(predicted_class, text):
    weight = {0:1, 1:8.5, 2:6.5}  
    base_weight = weight.get(predicted_class, 1)
    if "ร้ายแรง" in text or "จำนวนมาก" in text:
        base_weight = min(base_weight * 1.5, 10)
    return base_weight
text = "การละเมิดสิทธิบัตรที่ร้ายแรงและมีจำนวนมาก"
weight = get_physic_gate_preview(1, text)

print(f"text: {text}")
print(f"Predicted class: {1}, Weight: {weight}/10")
# 4 การสร้าง Json
import json
from datetime import datetime
def create_json_entry(doc_id, text):
    label = detect_category(text)
    conf, signals = cal_confidence(text,label)
    weight = get_physic_gate_preview(label,text)
    entry = {
        "id" : f"LAW-{doc_id:04d}",
        "text": text,
        "label": label,
        "metadata":{
            "confidence": conf,
            "context-signals": signals,
            "physic_gate_weight": weight,
            "processed_at": datetime.now().isoformat(),
            "requires_expert_review": conf <0.85
        }
    }
    return entry
# รัน Code  แสดงตัวอย่าง JSON
sample_entry = create_json_entry(1,"ละเมิดสิทธิบัตรการประดิษฐ์")
print(json.dumps(sample_entry, indent=4,ensure_ascii=False))