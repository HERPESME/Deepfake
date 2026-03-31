# How to Add Images to LaTeX Report in Overleaf

## Step 1: Prepare Images
All visualization images have been copied to the `figures/` directory. The following images are available:

### Celeb-DF Results:
- `efficientnet_b0_celebd_confusion_matrix.png`
- `efficientnet_b0_celebd_roc_curve.png`
- `xception_celebd_confusion_matrix.png`
- `xception_celebd_roc_curve.png`
- `resnet50_celebd_confusion_matrix.png`
- `resnet50_celebd_roc_curve.png`
- `vit_celebd_confusion_matrix.png`
- `vit_celebd_roc_curve.png`

### FaceForensics++ Results:
- `efficientnet_b0_faceforensics_confusion_matrix.png`
- `xception_faceforensics_confusion_matrix.png`
- `resnet50_faceforensics_confusion_matrix.png`
- `vit_faceforensics_confusion_matrix.png`

### Grad-CAM Visualizations:
- `efficientnet_b0_gradcam_fake.png`
- `efficientnet_b0_gradcam_real.png`
- `xception_gradcam_fake.png`
- `xception_gradcam_real.png`

## Step 2: Upload to Overleaf

1. **Create a folder in Overleaf:**
   - In your Overleaf project, click "New Folder" button
   - Name it `figures`

2. **Upload images:**
   - Click on the `figures` folder
   - Click "Upload" button
   - Select all PNG files from your local `figures/` directory
   - Wait for upload to complete

3. **Alternative: Upload via ZIP:**
   - Create a ZIP file of the `figures/` directory
   - In Overleaf, click "Upload" → "From .zip"
   - Select your ZIP file
   - Overleaf will extract it automatically

## Step 3: Verify Image Paths

The LaTeX file uses paths like:
```latex
\includegraphics[width=0.24\textwidth]{figures/efficientnet_b0_celebd_confusion_matrix.png}
```

Make sure:
- The `figures/` folder exists in your Overleaf project
- Image filenames match exactly (case-sensitive)
- Images are PNG format

## Step 4: Compile

1. Click "Recompile" in Overleaf
2. If images don't appear, check:
   - File paths are correct
   - Images are uploaded to the right folder
   - Image filenames match exactly (including case)

## Troubleshooting

**Images not showing?**
- Check the compilation log for errors
- Verify image paths are relative to the .tex file location
- Ensure images are in PNG format
- Check that filenames match exactly (no extra spaces)

**Images too large/small?**
- Adjust `width=0.24\textwidth` parameter
- For single column: use `width=0.48\textwidth`
- For full width: use `width=\textwidth`

**Compilation errors?**
- Make sure `graphicx` package is included (already in the file)
- Check for special characters in filenames
- Ensure all images are uploaded





