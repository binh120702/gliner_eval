from gliner import GLiNER
m = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")  # ví dụ model HF
labels = ["Person", "Location", "Organization"]
txt = "Ông Nguyễn Phú Trọng làm việc tại Hà Nội; tôi ở TP.HCM."

# Giữ greedy gốc:
ents0 = m.predict_entities(txt, labels, threshold=0.5)

# NMS: ổn định biên, giảm trùng lắp (flat hoặc nested đều OK)
ents1 = m.predict_entities(txt, labels, threshold=0.5,
                           decoding_algo="nms", iou_threshold=0.5)

# MWIS: tối ưu tổng thể cho FLAT (single-label)
ents2 = m.predict_entities(txt, labels, threshold=0.5,
                           flat_ner=True, multi_label=False, decoding_algo="mwis")
