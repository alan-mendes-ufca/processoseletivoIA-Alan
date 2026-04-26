# 🔢 Classificador de Dígitos MNIST — Edge AI

## Identificação do Candidato

- **Nome completo:** Alan Mendes Vieira
- **Github**: https://github.com/alan-mendes-ufca
---

## 1. 🗺️ Visão Geral da Solução

O projeto implementa uma CNN leve para classificar dígitos manuscritos do MNIST (0–9). O modelo é treinado em `train_model.py`, convertido e otimizado em `optimize_model.py`, e o resultado final é um arquivo `.tflite` compacto, pronto para rodar em dispositivos Edge com recursos limitados.

Fluxo do pipeline:
`TREINAR → SALVAR (.h5) → CONVERTER → QUANTIZAR → EXPORTAR (.tflite)`

---

## 2. 🏗️ Arquitetura do Modelo

O modelo recebe imagens `28×28×1` e passa por três convoluções compactas antes da classificação final. Sem camadas densas intermediárias — isso é o que mantém o número de parâmetros baixo.

**Camadas:**

- `Conv2D(16, 3×3, relu, padding="same")`
- `Conv2D(32, 3×3, relu, padding="same")`
- `MaxPooling2D(2×2)`
- `Conv2D(32, 3×3, relu, padding="same")`
- `MaxPooling2D(2×2)`
- `Flatten`
- `Dense(10, softmax)`

---

## 3. 🔌 Bibliotecas Utilizadas

- **Python 3.10 / 3.11** — CI validado em Python 3.10
- **tensorflow 2.21.0** — treinamento, conversão e quantização
- **numpy 2.4.4** — manipulação de dados

---

## 4. 🧠 Técnica de Otimização

A conversão para TensorFlow Lite usa **Dynamic Range Quantization**, aplicada em `optimize_model.py`:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

> **OBS**: Essa técnica converte os pesos de float32 para int8 dinamicamente, sem precisar de dataset de calibração. O resultado é um modelo muito mais leve, adequado para deploy em Edge AI.

---

## 5. ✅ Resultados Obtidos

| Métrica                | Valor      |
| ---------------------- | ---------- |
| Acurácia no teste      | **98,99%** |
| Parâmetros totais      | 29.738     |
| Épocas de treinamento  | 5          |
| Tamanho `model.h5`     | 388 KB     |
| Tamanho `model.tflite` | 35 KB      |
| Redução de tamanho     | **~91%**   |
