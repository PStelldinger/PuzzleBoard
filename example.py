import cv2
from puzzle_board.puzzle_board_detector import detect_puzzleboard


img = cv2.imread('examples//example1.png')

ids, coords = detect_puzzleboard(img.copy())

for i in range(len(ids)):
    py, px = ids[i]
    coordy, coordx = coords[i]
    font = cv2.FONT_HERSHEY_PLAIN
    s=str(px)
    textsize = cv2.getTextSize(s, font, 1.2, 2)[0]
    cv2.putText(img, s, (int(coordx - textsize[0] / 2), int(coordy - 1)), font, 1.2, (0,255,0), 2)
    s=str(py)
    textsize = cv2.getTextSize(s, font, 1.2, 2)[0]
    cv2.putText( img, s, (int(coordx - textsize[0] / 2), int(coordy + textsize[1] + 1)), font, 1.2, (0,255,0), 2)

cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
