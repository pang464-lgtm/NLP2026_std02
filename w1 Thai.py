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
    
#import deepcut
#print(f"deepcut.tokenize(test_text))
