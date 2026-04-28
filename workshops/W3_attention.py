import numpy as np
class SinusoidalPositionEncoding:
    def __init__(self, max_seq_len = 10, d_model = 16):
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(max_seq_len).reshape(-1,1)
        div = np.power(10000, np.arange(0, d_model, 2)/d_model)
        pe[:, 0::2] = np.sin(position / div)
        pe[:, 1::2] = np.cos(position / div)
        self.pe = pe
    def show(self,seq_len=5):
        print(f"Position Encoding (first {seq_len} tokwns):")
        for p in range(seq_len):
            print(f"Pos : {p}"+" ".join(f"{v:.2f}" for v in self.pe[p,:10])+"...")
    def show_with_words(self, words):
        print(f"Position Encoding with Words:")
        for p, w in enumerate(words):
            print(f"{p:<7} {w:<10} {self.pe[p,:10]}")
            for i,word in enumerate(words):
                if i >= len(self.pe):
                    break
                vec = self.pe[i,:4]
                vec_str = " ".join(f"{v:.3f}" for v in vec)
                print(f"pos {i:<3} word '{word}' : {vec_str}")
#การใช้งาน
#words = ["ฉัน","รัก","ภาษา","ไทย","เรียน","รู้","จัก","การ","เข้ารหัส","ตำแหน่ง"]
words_list = ["I","Love","thai","learning","know","about","position","encoding"]
pe = SinusoidalPositionEncoding(max_seq_len=10, d_model=16)
#pe.show_with_words(words)
pe.show_with_words(words_list)
#pe = SinusoidalPositionEncoding()
#pe.show()

# 2. scale dot-product and padding mask ความยาวของเอกสารเท่ากัน
def scale_dot_product(q,k,v,mask=None):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0,2,1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.matmul(weights, v)
    return output, weights
#จำลองข้อมูล query, key, value และทดสอบฟังก์ชัน attention
np.random.seed(2)
q = k = v = np.random.rand(1,3,6)
output, weights = scale_dot_product(q, k, v)
print(f"Attention Weights:\n{weights}")
print(weights[0].round(3))