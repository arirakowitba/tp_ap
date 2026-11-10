import json
with open('results/item_label_confidences_summary.json', encoding='utf-8') as f:
    d = json.load(f)

top50 = d['top_50_highest_confidence_corrections']
print("Inspecting Top 20 corrections from deconvolution summary:")
for i, item in enumerate(top50[:20]):
    print(f"{i+1:2d}. ID: {item['item_id']:5d} | Anchor: {item['anchor_label']} -> Rec: {item['recommended_label']} | Cue: {str(item.get('semantic_cue'))[:15]} | Cat: {item['category_id']} | Price: {item['price']} | Title: {item['title'][:45]}")
