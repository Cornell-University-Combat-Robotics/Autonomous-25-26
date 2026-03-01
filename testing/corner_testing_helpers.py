import math
import cv2

def draw_orientation_arrow(frame, detected_bots_with_data, arrow_length=50, thickness=2):
    for bot_name, data in detected_bots_with_data.items():
        if not data or 'orientation' not in data or 'bbox' not in data:
            continue

        x, y, w, h = data['bbox']
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        angle_rad = math.radians(data['orientation'])
        ex = int(cx + arrow_length * math.cos(angle_rad))
        ey = int(cy - arrow_length * math.sin(angle_rad)) 

        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), thickness, tipLength=0.3)
        cv2.putText(frame, f"{bot_name}: {data['orientation']:.1f}°",
                    (cx + 5, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
