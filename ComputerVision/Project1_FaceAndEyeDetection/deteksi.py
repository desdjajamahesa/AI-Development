import cv2
import sys

def main():
  print("Memulai Program Deteksi Wajah dan Mata !")
  print("Tekan 'Q' untuk keluar dari program")

  try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if face_cascade.empty():
      print("ERROR : File haarcascade_frontalface_default.xml Tidak Dapat Dimuat !")
      return
    if eye_cascade.empty():
      print("ERROR : File haarcascade_eye.xml Tidak Dapat Dimuat !")
      return

    print("Model Cascade Berhasil Dimuat!")

  except Exception as e:
    print(f'ERROR SAAT MEMBUAT CASCADE : {e}')
    return

  try:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
      print("ERROR : Tidak Bisa Akses Kamera")
      print("Pastikan Kamera Terhubung dan Tidak Digunakan Pada Aplikasi Lain !")
      return

    print("Kamera Berhasil Diakses!")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

  except Exception as e:
    print(f"ERROR SAAT AKSES KAMERA : {e}")
    return

  while True:
    ret, frame = cap.read()

    if not ret:
      print("ERROR : TIDAK BISA MEMBACA FRAME DARI KAMERA")
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

      cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

      roi_gray = gray[y:y+h, x:x+w]
      roi_color = frame[y:y+h, x:x+w]

      eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
      )

      for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.putText(roi_color, 'Eye', (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.putText(frame, f'Wajah Terdeteksi : {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, 'Tekan Q Untuk Keluar', (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Face and Eye Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
      print("Program Dihentikan Oleh User")
      break

    if cv2.getWindowProperty('Face and Eye Detection', cv2.WND_PROP_VISIBLE) < 1:
      break

  print("Membersihkan Resource...")
  cap.release()
  cv2.destroyAllWindows()
  print("Program Selesai !")

if __name__ == "__main__":
  main()