# 项目文件清理报告

## 📋 可以删除的文件

### 1. **Python缓存文件** 🗑️ 建议删除
```
__pycache__/                    # 根目录Python缓存
lsc/__pycache__/                # lsc模块缓存
```
**原因**：运行时自动生成，可以安全删除

---

### 2. **旧的增益表测试文件** 🗑️ 建议删除
```
gain_table_to_invert.txt        # 904 bytes
inverted_gain_table.txt         # 1685 bytes
invert.py                       # 3425 bytes
```
**原因**：这些是早期测试文件，现在已经不需要了

---

### 3. **重复的优化文档** 🗑️ 建议删除
```
V3.2_OPTIMIZATION.md            # 5064 bytes
```
**原因**：内容已经整合到 `CHANGELOG_V3.2.md`，重复了

---

### 4. **分析脚本** ⚠️ 可选删除
```
analyze_shading.py              # 6079 bytes
```
**原因**：如果不再需要单独分析暗角，可以删除

---

### 5. **XML配置文件** ⚠️ 可选保留
```
lsc34_bps.xml                   # 21356 bytes
```
**原因**：高通ISP配置参考文件，建议保留

---

## ✅ 应该保留的文件

### 核心代码
- ✅ `main.py` - 主程序
- ✅ `config.py` - 配置文件
- ✅ `lsc/` - 核心模块

### 文档
- ✅ `README.md` - 项目说明
- ✅ `CHANGELOG_V3.2.md` - 最新更新日志
- ✅ `QUALCOMM_LSC_COMPARISON.md` - 高通对比分析
- ✅ `OPTIMIZATION_RECOMMENDATIONS.md` - 优化建议
- ✅ `ALGORITHM.md` - 算法说明
- ✅ `V3_PANORAMA_OPTIMIZATION.md` - V3.0全景优化说明
- ✅ `V3.1_RELEASE_NOTES.md` - V3.1发布说明

### 参考资料
- ✅ `lsc.pdf` - 高通LSC技术文档（重要参考）
- ✅ `lsc34_bps.xml` - 高通ISP配置参考

### 其他
- ✅ `requirements.txt` - 依赖列表
- ✅ `LICENSE` - 开源协议
- ✅ `.gitignore` - Git忽略规则
- ✅ `input/` - 输入图像目录
- ✅ `output/` - 输出结果目录

---

## 🧹 清理建议

### 方案1：保守清理（推荐）
只删除缓存文件和明确无用的文件：
```bash
# 删除Python缓存
rm -rf __pycache__ lsc/__pycache__

# 删除旧测试文件
rm gain_table_to_invert.txt inverted_gain_table.txt invert.py

# 删除重复文档
rm V3.2_OPTIMIZATION.md
```

### 方案2：彻底清理
额外删除分析脚本：
```bash
# 方案1的所有操作 +
rm analyze_shading.py
```

---

## 📊 清理后效果

| 项目 | 清理前 | 清理后 | 节省 |
|------|--------|--------|------|
| 文件数量 | ~30个 | ~25个 | -5个 |
| 磁盘空间 | ~3MB | ~3MB | ~10KB |
| 项目整洁度 | 中等 | 优秀 | ✅ |

---

## ⚠️ 注意事项

1. **不要删除 `.git/` 目录** - 这是Git版本控制数据
2. **不要删除 `.venv/` 目录** - 这是Python虚拟环境
3. **不要删除 `output/` 目录** - 包含校准结果
4. **备份重要数据** - 删除前确认不需要

---

**建议**：执行方案1（保守清理），保持项目整洁的同时不丢失任何有用信息。
