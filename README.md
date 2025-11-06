# 🌊 Segmentação Semântica de [Manchas de Óleo] em Imagens Marítimas

## 💡 Descrição do Projeto (APS)

Este projeto foi desenvolvido como **APS (Atividade Prática Supervisionada)** para o estudo e aplicação de técnicas avançadas de **Visão Computacional** e **Machine Learning** (Deep Learning).

O objetivo principal é a **detecção e segmentação semântica** de **[Manchas de Óleo/Algas Nocivas/etc.]** em imagens de superfície. O sistema integra um pipeline robusto que combina técnicas clássicas de **Processamento Digital de Imagens (PDI)** para pré-processamento (CLAHE, Filtro Bilateral) com uma **Rede Neural Convolucional (CNN)** baseada na arquitetura **U-Net** para a segmentação precisa pixel a pixel.

### 🌟 Diferencial

O diferencial do projeto reside na utilização da arquitetura **U-Net**, especializada em segmentação, para gerar uma **máscara binária** que isola a área de interesse, garantindo maior precisão em relação a métodos puramente baseados em limiarização.

---

## 🎯 Componentes e Funcionalidades Chave

O projeto é dividido em dois módulos principais: **Treinamento da CNN** e **Pipeline de Inferência/PDI**.

### Módulo de Treinamento (U-Net)

* **Arquitetura:** Implementação da **U-Net** com *Skip Connections* para preservar informações de contexto e detalhes espaciais.
* **Segmentação:** Focado em segmentação binária (**óleo vs. água**).
* **Compilação:** Utiliza `binary_crossentropy` (ideal para segmentação binária) e métricas como `MeanIoU` (Intersection over Union).
* **Conjunto de Dados:** Utiliza dados simulados (`generate_simulated_data`) para fins de demonstração da estrutura.

### Módulo de Inferência e PDI

| Etapa | Técnica | Biblioteca | Objetivo |
| :---: | :---: | :---: | :--- |
| **1. Pré-processamento** | Filtro Bilateral | `cv2` | Redução de ruído preservando as bordas. |
| **2. Realce de Contraste** | CLAHE (Equalização Adaptativa) | `skimage.exposure` | Aumentar a visibilidade de áreas escuras (manchas de óleo) no fundo. |
| **3. Segmentação (IA)** | Inferência U-Net | `tensorflow` | Geração da máscara binária predita pela CNN. |
| **4. Pós-processamento** | Morfologia (Fechamento, Dilatação) | `skimage.morphology` | Remoção de ruído isolado e preenchimento de pequenos buracos na máscara. |
| **5. Visualização** | Contornos | `cv2` | Desenho de contornos verdes sobre a imagem original para destaque visual da área detectada. |

---

## 💻 Tecnologias e Bibliotecas Utilizadas

| Categoria | Tecnologia | Uso Principal |
| :---: | :---: | :--- |
| **Linguagem** | **Python 3.x** | Linguagem de desenvolvimento principal. |
| **Deep Learning** | **TensorFlow/Keras** | Construção, treinamento e inferência do modelo U-Net. |
| **Visão Computacional** | **OpenCV (`cv2`)** | Carregamento de imagens e detecção/desenho de contornos. |
| **PDI e Matemática** | **NumPy, scikit-image** | Manipulação eficiente de arrays, CLAHE e operações morfológicas. |
| **Visualização** | **Matplotlib** | Plotagem e comparação dos resultados (máscaras, imagens realçadas e finais). |

---

## 🛠️ Como Executar o Projeto Localmente

### Pré-requisitos

Certifique-se de ter o **Python 3.8+** e o **`pip`** instalados.

### 1. Instalação

1.  **Clone o Repositório:**
    ```bash
    git clone [LINK DO SEU REPOSITÓRIO]
    cd [pasta-do-projeto]
    ```

2.  **Instale as Dependências:**
    Crie o arquivo `requirements.txt` (se ainda não tiver) com as bibliotecas:
    ```
    tensorflow
    opencv-python
    numpy
    matplotlib
    scikit-image
    ```
    E execute:
    ```bash
    # (Opcional) Ative seu ambiente virtual
    # source venv/bin/activate
    
    pip install -r requirements.txt
    ```

### 2. Preparação dos Dados

* **Para Treinamento:** O código utiliza dados simulados. Para treinar o modelo com dados reais, substitua a função `generate_simulated_data` pelo carregamento do seu dataset real (imagens e suas *ground truth masks*).
* **Para Inferência:** Crie a pasta e coloque as imagens de teste:
    ```
    # Crie uma pasta 'imagens_teste' e coloque os arquivos JONAS*.jpg nela
    mkdir imagens_teste
    # Mova as imagens de teste:
    mv JONAS2.jpg JONAS3.jpg JONAS4.jpg imagens_teste/
    ```
    *Ajuste a lista `lista_de_imagens` no código de inferência se necessário.*

### 3. Ordem de Execução

**A. Treinar o Modelo:**

1.  Execute o código de treinamento da U-Net. Este passo gerará o arquivo `modelo_segmentacao_oleo.h5`.
    ```bash
    python treino_unet.py # Se você separou o código em dois arquivos
    ```

**B. Executar o Pipeline de Inferência/PDI:**

1.  Certifique-se de que o arquivo `modelo_segmentacao_oleo.h5` está na mesma pasta.
2.  Execute o código principal de processamento.
    ```bash
    python pipeline_segmentacao.py # Se você separou o código em dois arquivos
    ```
*O script irá iterar pelas imagens na lista, aplicando o pipeline de PDI, chamando a CNN para segmentação e exibindo 4 gráficos de resultado para cada imagem.*

---
