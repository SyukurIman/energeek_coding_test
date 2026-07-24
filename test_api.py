"""
Test Client untuk AI Service API
Mengirim gambar ke API dan menampilkan hasil JSON.
"""

import requests
import json
import sys
import os


def test_predict(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Kirim gambar ke endpoint /predict dan tampilkan hasilnya.
    
    Args:
        image_path: Path ke file gambar
        api_url: Base URL dari AI Service
    """
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        return None
    
    url = f"{api_url}/predict"
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        
        print(f"📤 Mengirim {image_path} ke {url}...")
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Hasil Analisis:")
        print(f"   {json.dumps(result, indent=4)}")
        return result
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None


def test_health(api_url: str = "http://localhost:8000"):
    """Cek health status service."""
    url = f"{api_url}/health"
    response = requests.get(url)
    
    if response.status_code == 200:
        print(f"✅ Service OK: {response.json()}")
    else:
        print(f"❌ Service error: {response.status_code}")
    
    return response.json() if response.status_code == 200 else None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test client untuk AI Service API")
    parser.add_argument("image", nargs="?", help="Path ke file gambar yang akan dianalisis")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL API (default: http://localhost:8000)")
    parser.add_argument("--health", action="store_true", help="Cek health status saja")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("  Energeek - API Test Client")
    print("=" * 50)
    
    if args.health:
        test_health(args.url)
    elif args.image:
        test_predict(args.image, args.url)
    else:
        # Interactive mode
        print("\nPilih opsi:")
        print("1. Cek health status")
        print("2. Kirim gambar untuk analisis")
        
        choice = input("\nPilihan (1/2): ").strip()
        
        if choice == "1":
            test_health(args.url)
        elif choice == "2":
            path = input("Path gambar: ").strip()
            if path:
                test_predict(path, args.url)
            else:
                print("❌ Path gambar tidak boleh kosong")
        else:
            print("❌ Pilihan tidak valid")
