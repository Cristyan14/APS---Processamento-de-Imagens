# FLUXO PRINCIPAL: Processando uma Lista de Imagens
# ----------------------------------------------------------------------

# Lista das imagens a serem processadas
lista_de_imagens = ["JONAS2.jpg", "JONAS3.jpg", "JONAS4.jpg"]

for nome_arquivo in lista_de_imagens:
    print(f"\n========================================================")
    print(f"PROCESSANDO IMAGEM: {nome_arquivo}")
    print(f"========================================================")

    # 1. Carregar imagem
    img = cv2.imread(nome_arquivo)
    if img is None:
        print(f"ERRO: Imagem '{nome_arquivo}' não encontrada. Pulando para a próxima.")
        continue # Pula para a próxima imagem da lista

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Redução de ruído (PDI)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

    # 3. Equalização adaptativa (PDI)
    equalized = exposure.equalize_adapthist(denoised, clip_limit=0.02)
    equalized = (equalized * 255).astype(np.uint8)

    # 4. SEGMENTAÇÃO COM IA/CNN
    mask_from_cnn = load_and_predict_cnn(equalized, img.shape)

    # 5. PÓS-PROCESSAMENTO MORFOLÓGICO (Menos agressivo, mais focado)
    print("Aplicando ajustes morfológicos...")
    mask = morphology.remove_small_objects(mask_from_cnn, min_size=500)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.binary_dilation(mask, morphology.disk(2))
    mask = morphology.remove_small_objects(mask, min_size=800)

    # 6. Converter máscara para formato OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 7. Encontrar e desenhar contornos
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8. Desenhar contornos verdes e preenchimento leve
    img_contoured = img.copy()
    overlay = img.copy()

    for cnt in contours:
        cv2.drawContours(img_contoured, [cnt], -1, (0, 255, 0), 3)
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), -1)

    # Efeito de transparência para visual bonito
    highlight = cv2.addWeighted(overlay, 0.3, img_contoured, 0.7, 0)

    # 9. Exibir resultados
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    # Título principal para cada imagem
    fig.suptitle(f"Resultado da Detecção - Arquivo: {nome_arquivo}", fontsize=14)

    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Original (Cinza)")
    axes[1].imshow(equalized, cmap="gray"); axes[1].set_title("Equalizada (CLAHE)")
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title("Máscara Final (Segmentação CNN)")
    axes[3].imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB)); axes[3].set_title("Manchas Destacadas (Contorno Verde)")

    for ax in axes: ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste para caber o título
    plt.show()

print("\nProcessamento de todas as imagens concluído.")
