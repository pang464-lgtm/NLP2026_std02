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
        base_confidence = 0.7
        signal.append("statotury_reference")
        
    if "คำพิพากษา" in text or "ศาล" in text:
        base_confidence = 0.8
        signal.append("precedent_reference")
text_with_statute = "ละเมิดสิทธิบัตรตามมาตรา 123 ของพ.ร.บ.สิทธิบัตร"
conf , sig = cal_confidence(text_with_statute, "patent")
print(f"text: {text_with_statute}")
print(f"Confidence: {conf}, Signals: {sig}")   
print(f"Predicted class: patent, Confidence: {conf}, Signals: {sig}")