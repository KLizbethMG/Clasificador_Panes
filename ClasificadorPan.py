import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import flet as ft
import base64

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

IMG_SIZE = 64
FONDOS = ["IM_F_BLANCO", "IM_F_COLOR"]

modelos_produccion = {}
nombres_modelos_prod = {}
metricas_globales = {}
clases = {0: "CONCHA", 1: "OJO"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel() 
grafica_idx = 0

for fondo in FONDOS:
    print(f"\n=============================================")
    print(f" PROCESANDO EXPERIMENTO: {fondo}")
    print(f"=============================================")
    
    ruta_carpeta = os.path.join("dataset", fondo)
    X = []
    y = []
    
    if not os.path.exists(ruta_carpeta):
        print(f"Alerta: No se encontró la carpeta real '{ruta_carpeta}'")
        continue
        
    conteo_conchas = 0
    conteo_ojos = 0
    
    for archivo in os.listdir(ruta_carpeta):
        archivo_minusculas = archivo.lower()
        
        if archivo_minusculas.endswith(('.png', '.jpg', '.jpeg', '.webp', '.lepg', '.jepg')):
            img_path = os.path.join(ruta_carpeta, archivo)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if "concha" in archivo_minusculas:
                X.append(img.flatten() / 255.0)
                y.append(0) 
                conteo_conchas += 1
            elif "ojo" in archivo_minusculas:
                X.append(img.flatten() / 255.0)
                y.append(1) 
                conteo_ojos += 1

    print(f"-> Éxito en {fondo}: Se detectaron {conteo_conchas} Conchas y {conteo_ojos} Ojos por su nombre.")
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print(f"ERROR: No se pudieron extraer imágenes válidas de '{ruta_carpeta}'.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo_lineal = SVC(kernel='linear', probability=True, random_state=42)
    modelo_rbf    = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    
    modelo_lineal.fit(X_train, y_train)
    modelo_rbf.fit(X_train, y_train)
    
    acc_lineal = accuracy_score(y_test, modelo_lineal.predict(X_test))
    acc_rbf    = accuracy_score(y_test, modelo_rbf.predict(X_test))
    
    print(f"-> Accuracy SVM Lineal: {acc_lineal:.4f}")
    print(f"-> Accuracy SVM RBF:    {acc_rbf:.4f}")
    
    if acc_rbf >= acc_lineal:
        modelos_produccion[fondo] = modelo_rbf
        nombres_modelos_prod[fondo] = "SVM RBF"
    else:
        modelos_produccion[fondo] = modelo_lineal
        nombres_modelos_prod[fondo] = "SVM Lineal"
        
    metricas_globales[fondo] = f"Lineal: {acc_lineal:.2f} | RBF: {acc_rbf:.2f}"
    
    print(f"Ganador {fondo}: {nombres_modelos_prod[fondo]}")
    print(classification_report(y_test, modelos_produccion[fondo].predict(X_test), target_names=["CONCHA", "OJO"]))

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    X_train_2d = pca.transform(X_train)
    
    svm_lineal_2d = SVC(kernel='linear').fit(X_train_2d, y_train)
    svm_rbf_2d    = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train_2d, y_train)
    
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    ax_l = axes[grafica_idx]
    Z_l = svm_lineal_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax_l.contourf(xx, yy, Z_l, alpha=0.3, cmap='coolwarm')
    ax_l.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1], c='royalblue', edgecolors='k', s=30)
    ax_l.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1], c='tomato', edgecolors='k', s=30)
    ax_l.set_title(f"{fondo} - SVM Lineal (Acc: {acc_lineal:.2f})")
    ax_l.grid(True)
    grafica_idx += 1
    
    ax_r = axes[grafica_idx]
    Z_r = svm_rbf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax_r.contourf(xx, yy, Z_r, alpha=0.3, cmap='coolwarm')
    ax_r.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1], c='royalblue', edgecolors='k', s=30, label='CONCHA')
    ax_r.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1], c='tomato', edgecolors='k', s=30, label='OJO')
    ax_r.set_title(f"{fondo} - SVM RBF (Acc: {acc_rbf:.2f})")
    ax_r.grid(True)
    if grafica_idx == 1: ax_r.legend(loc='upper right')
    grafica_idx += 1

if grafica_idx > 0:
    plt.suptitle('Estudio Comparativo SVM — Impacto del Fondo en el Clasificador', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("comparacion_completa_fondos.png", dpi=150)
    print("\nSúper gráfica de 4 paneles guardada como 'comparacion_completa_fondos.png'")

# ==================== CAPA DE INTERFAZ MÓVIL (FLET) ====================

def main(page: ft.Page):
    page.title = "Pan Dulce Classifier Mobile"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = "adaptive"

    page.window_width = 400       
    page.window_height = 750      
    page.window_resizable = False 

    resultado_txt = ft.Text(
        "Selecciona o toma una foto para clasificar", 
        size=16, 
        weight=ft.FontWeight.BOLD,
        text_align=ft.TextAlign.CENTER
    )
    
    preview_img = ft.Image(
        src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuRyTfseqkAYTVvEmgPhnrXf4jIb5RmfW_ww&s",
        width=250,
        height=250,
        fit="contain",
    )

    def realizar_prediccion_bytes(img_bytes):
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            resultado_txt.value = "Error al decodificar los datos de la imagen."
            page.update()
            return
            
        img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        img_flat = img_gray.flatten().reshape(1, -1) / 255.0
        
        pred_b = modelos_produccion["IM_F_BLANCO"].predict(img_flat)[0] if "IM_F_BLANCO" in modelos_produccion else None
        pred_c = modelos_produccion["IM_F_COLOR"].predict(img_flat)[0] if "IM_F_COLOR" in modelos_produccion else None
        
        res_b = clases[pred_b] if pred_b is not None else "No entrenado"
        res_c = clases[pred_c] if pred_c is not None else "No entrenado"
        
        resultado_txt.value = (
            f"[FONDO BLANCO]: {res_b} ({nombres_modelos_prod.get('IM_F_BLANCO', 'N/A')})\n\n"
            f"[FONDO COLOR]: {res_c} ({nombres_modelos_prod.get('IM_F_COLOR', 'N/A')})"
        )
        page.update()

    def procesar_archivo_seleccionado(e: ft.FilePickerResultEvent):
        if e.files:
            foto = e.files[0]
            
            if e.files[0].path is None:
                upload_url = page.get_upload_url(foto.name, 600)
                picker_galeria.upload_files([ft.FilePickerUploadFile(foto.name, upload_url)])
                return
                
            if os.path.exists(foto.path):
                with open(foto.path, "rb") as f:
                    img_bytes = f.read()
                base64_img = base64.b64encode(img_bytes).decode("utf-8")
                preview_img.src = None
                preview_img.src_base64 = base64_img
                page.update()
                realizar_prediccion_bytes(img_bytes)

    def on_upload_complete(e: ft.FilePickerUploadEvent):
        ruta_temporal = os.path.join("uploads", e.file_name)
        if os.path.exists(ruta_temporal):
            with open(ruta_temporal, "rb") as f:
                img_bytes = f.read()
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            preview_img.src = None
            preview_img.src_base64 = base64_img
            page.update()
            realizar_prediccion_bytes(img_bytes)
            try:
                os.remove(ruta_temporal) 
            except:
                pass

    picker_galeria = ft.FilePicker(on_result=procesar_archivo_seleccionado)
    picker_galeria.on_upload_progress = on_upload_complete
    
    page.overlay.append(picker_galeria)

    page.add(
        ft.AppBar(title=ft.Text("App - Analizador de Panes"), bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST),
        ft.Container(height=20), 
        preview_img,
        ft.Container(height=20), 
        
        ft.Column(
            controls=[
                ft.ElevatedButton(
                    "Subir de Galería",
                    icon=ft.Icons.IMAGE,
                    width=280,
                    on_click=lambda _: picker_galeria.pick_files(
                        allow_multiple=False, 
                        file_type=ft.FilePickerFileType.IMAGE
                    )
                ),
                ft.ElevatedButton(
                    "Tomar Fotografía",
                    icon=ft.Icons.CAMERA_ALT,
                    width=280,
                    on_click=lambda _: picker_galeria.pick_files(
                        allow_multiple=False, 
                        file_type=ft.FilePickerFileType.IMAGE,
                        camera_capture=True
                    )
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10
        ),
        
        ft.Card(
            content=ft.Container(
                content=resultado_txt,
                padding=20
            ),
            margin=20
        )
    )

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    ft.app(target=main, port=8080, upload_dir="uploads")
