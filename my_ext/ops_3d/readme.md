# 通用3D操作

## 内容:

- SO3: 旋转变换
- SE3: 刚性变换 (旋转+平移)
- affine: 仿射变换
- 坐标系: 各种坐标系之间的变换
- shading 着色器
- light 光照

## 标准

采用与OpenGL相同的约定

- World Space 世界坐标系(右手系): y轴向上, x轴向右, z轴向外
- View Space/Camera Space 观察坐标系, 同世界坐标系
- Clip Spave 裁剪坐标系(左手系): y轴向上, x轴向右, z轴向里, (z 的坐标值越小，距离观察者越近 )
- 屏幕坐标系： X 轴向右为正，Y 轴向下为正，坐标原点位于窗口的左上角 (左手系: z轴向屏幕内，表示深度)
- 矩阵乘法: 左乘, 如: p'=TRp 表示点p先旋转再平移
- 存储方法: 列主序, 矩阵

```text
r1 r2 r3 tx
r4 r5 r6 ty
r7 r8 r9 tz
0  0  0  1
```

存储为: `[r1 r4 f7 0; r2 r5 r8 0; r3 r6 r9 0; tx ty tz 1]`

原因: SIMD并行化 p' = p * col1 + p * ocl2 + p * col3 + col4

- CPU基于glm实现