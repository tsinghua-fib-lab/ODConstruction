# Intro
## story
- OD生成任务的实际意义
- 现有OD工作的局限性
  - 知识驱动
    - 简单、建模能力差 但 universal
    - 数据弥补知识局限，但数据少、参数少
    - non-global
  - 数据驱动 ml
    - 建模能力强 但 有bias，泛化不好
    - 缺乏知识的指导
    - quasi-global
- 数据驱动与知识驱动相结合的一些工作取得了成功（问寰东师兄）
  - 数据拟合物理公式参数
  - 知识约束模型参数
- 挑战
  - physics（公式） 与 data（embedding） 形式之间存在gap，难以直接结合
  - 现有的graph learning有些局限性
  - global 难以利用知识评价
- 方法提出
  - embedding 在学习时，接受知识的约束，知识结合数据
  - multi-graph 的 弥补
  - GAN，global considering learning

## contribution
- 提出将data与physics结合，来解决OD生成问题
- 设计了data与physics结合的模型
  - 重力公式的引入、global considered
- 实验证明了模型的性能，证实了知识结合数据的可行性


# Preliminaries
## notations
## problem

# Methodology
- 数据数据协同驱动建模
  - encoder: graph embedding
  - decoder: gravity-formulated prediction
- 基于GAN的global considering training
  - generator
  - random walk sampling
  - discriminator

# Exp
## overall performance
## ablation
## explainable
- mass with indicator
## case
- visualization of Groundtruth、physics、data-driven、ours

# related work
- OD知识驱动的方法
- OD数据驱动的方法
