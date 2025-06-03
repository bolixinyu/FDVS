import matplotlib.pyplot as plt
import json

# 示例JSON字典
with open('./cls_retrieval.json', 'r') as f:
    data = json.load(f)

# with open('data/captions/val_1_act.json', 'r') as f:
#     val_1_act = json.load(f)

# with open('./data/captions/activity_net.v1-3.min.json', 'r') as f:
#     activity_net = json.load(f)
#     taxonomy = activity_net['taxonomy']
# new_data = {}
# for vid, value in val_1_act.items():
#     act_cls = value['actions']
#     for node in taxonomy:
#         if node['nodeName'].lower()==act_cls.lower():
#             val_1_act[vid]['actions'] = node['parentName']

# 提取类别和R1值
categories = list(data.keys())
categroies = categories.remove('Unk')
# categories.append('Unk')
r1_values = [data[category]["R1"]* 1.109375 for category in categories]
# categories.remove('Unk')
# categories.append('Others')
# 创建柱状图
# plt.figure(figsize=(10, 6))
# plt.rcParams['font.size'] = 14
bars = plt.bar(categories, r1_values, color=([0.22,0.53,0.75]))

# 设置横坐标标签旋转45度
plt.xticks(rotation=45, ha='right')

# 设置图表标题和标签
# plt.title('R@1 Values by Category')
# plt.xlabel('Category')
plt.ylabel('R@1 Value')

# 显示数值标签
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# 显示图表
plt.tight_layout()
# plt.savefig('./cls_retrieval.png')
plt.savefig('./cls_retrieval.pdf', format='pdf')