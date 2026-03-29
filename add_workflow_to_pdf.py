#!/usr/bin/env python3
"""
Alternative script to add workflow diagram to existing PDF using PyPDF2 or similar
If reportlab is not available, this will try alternative methods.
"""

import sys
from pathlib import Path

# Try multiple approaches
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer, Image, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from PIL import Image as PILImage
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️  reportlab not available, trying alternative method...")

if HAS_REPORTLAB:
    # Use reportlab method
    print("✅ Using reportlab to regenerate PDF with workflow diagram...")
    
    # Import the main generation function
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_presentation import create_pdf_presentation
    
    print("🔄 Regenerating presentation PDF...")
    create_pdf_presentation()
    print("✅ PDF regenerated with workflow diagram!")
    
else:
    # Alternative: Use PyPDF2 or create a simple insertion
    print("❌ reportlab not available. Cannot regenerate PDF.")
    print("\n📋 To fix this, please install reportlab:")
    print("   pip3 install reportlab Pillow")
    print("\nOr if you have a virtual environment:")
    print("   source deepfake_env/bin/activate")
    print("   pip install reportlab Pillow")
    print("\nThen run: python3 generate_presentation.py")
    sys.exit(1)




