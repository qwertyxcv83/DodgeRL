def rect_collide(centerX1, centerY1, centerX2, centerY2, width1, height1, width2, height2):
    return abs(centerX1-centerX2) < (width1 + width2)/2 and abs(centerY1-centerY2) < (height1 + height2)/2


def is_in_rect(x, y, centerx, centery, width, height):
    return abs(x-centerx) < width/2 and abs(y-centery) < height/2
