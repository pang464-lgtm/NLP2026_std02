import re

LEGAL_KEYWORD = [
        "สิทธิบัตรการประดิษฐ์", "ลิขสิทธิ์", "เครื่องหมายการค้า",
        "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ", "การประดิษฐ์ขึ้นใหม่",
        "ขั้นการประดิษฐ์", "ทางอุตสาหกรรม", "จำคุก", "ปรับ",
        "ริบทรัพย์", "เจ้าของสิทธิ", "ผู้ทรงสิทธิ", "คำขอรับสิทธิบัตร",
        "การดัดแปลง", "การเผยแพร่", "การทำซ้ำ", "พระราชบัญญัติ",
        "คณะกรรมการ", "พนักงานเจ้าหน้าที่", "อุทธรณ์",
        "ศาลทรัพย์สินทางปัญญา", "เครื่องหมายบริการ", "เครื่องหมายรับรอง",
        "เครื่องหมายการค้า"]

def legal_tokenizer(text):
    compound = '|'.join(map(re.escape,sorted(LEGAL_KEYWORD,key =len,reverse=True)))
    pattern = compound + r"|[\u0E00-\u0E7F]+" + r"|[a-zA-Z0-9]+"
    tokens = re.findall(pattern, text)
    return tokens   
    return re.findall(pattern, text)

test_text = "จำเลยกระทำการละเมิดสิทธิบัตรการประดิษฐ์ขึ้นใหม่โดยการดัดแปลงและเผยแพร่ผลงานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาตตามพระราชบัญญัติทรัพย์สินทางปัญญา"
tokens = legal_tokenizer(test_text)
print(f"input : {test_text}")
print(f"output : {tokens}")
    
#tokens_raw = word_toknerizer(protectde,test_text)
def calcilate_baseline_amviguity(text):
    matches = []
    for word in LEGAL_KEYWORD:
        for match in re.finditer(word, text):
            matches.append((match.start(), match.end(), word))
    overlap = 0
    for i in range(len(matches)):
        for j in range(i+1, len(matches)):
            if matches[i][0] < matches[j][1] and matches[i][1] > matches[j][0]:
                overlap += 1
    return overlap/len(matches) if matches else 0

sample_text = "จำเลยกระทำการละเมิดสิทธิบัตรการประดิษฐ์ขึ้นใหม่โดยการดัดแปลงและเผยแพร่ผลงานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาตตามพระราชบัญญัติทรัพย์สินทางปัญญา"
baesline_tokens = legal_tokenizer(sample_text)
baesline_rate = calcilate_baseline_amviguity(sample_text)
print(f"W1_Baseline Tokens : {baesline_tokens}")
print(f"token ambiguity rate : {baesline_rate}")

from transformers import AutoTokenizer
model_name = "ai4thai/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def berta_tokenizer(text):
        return tokenizer.tokenize(text)
        return [t.replace("▁","") for t in tokens if t.replace("▁","")]
def analyze_berta_ambiguity(text,legal_keywords =LEGAL_KEYWORDS):
        tokens = berta_tokenizer(text)
        frag_score = []
        for kw in legal_keywords:
            if kw in text:
                kw_tokens = berta_tokenizer(kw)
                fragment_ratio = len(kw_tokens)/1
                frag_score.append((kw,fragment_ratio))
        avg_frag = (sum(frag_score)/len(frag_score)) -1 if frag_score else 0
        return min(avg_frag,1.0)
#รัน แสดงผล Wangchanberta
refined_tokens = berta_tokenizer(sample_text)
refined_rate = analyze_berta_ambiguity(sample_text,LEGAL_KEYWORDS)  