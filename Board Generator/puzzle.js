
function generateCode(BASE_CODE, BASE_CODE2) {
    //console.log(BASE_CODE)
    lines = BASE_CODE.split("\n")
    lines2 = BASE_CODE2.split("\n")
    //console.log(lines)
    let H = lines.length
    if (H != lines2.length) alert("basecode rows do not match!");
    let W = lines[0].length
    if (W != lines2[0].length) alert("basecode cols do not match!");
    var CODE_BIG = []
    for (var y = 0; y < W * H; ++y) {
        for (var x = 0; x < W * H; ++x) {
            a = lines[y % H][x % W] == '1' ? 1 : 0
            b = lines2[x % H][y % W] == '1' ? 2 : 0
            CODE_BIG.push(a + b)
        }
    }
    //console.log(CODE_BIG)
    return [CODE_BIG, H * W]
}

function generatePattern(
    WIDTH
    , HEIGHT
    , START_X
    , START_Y
    , RES_PER_SQ
    , PAPER_H
    , PAPER_W
    , BASE_CODE
    , BASE_CODE2) {

    const [code2, CODE_W] = generateCode(BASE_CODE, BASE_CODE2)
    let offx = (PAPER_W - WIDTH * RES_PER_SQ) / 2
    let offy = (PAPER_H - HEIGHT * RES_PER_SQ) / 2
    //console.log(PAPER_W, PAPER_H)
    let ctx = new PDFCanvas("my.pdf", PAPER_W, PAPER_H)

    // generate checker board
    for (var x = 0; x < WIDTH + 1; ++x) {
        for (var y = 0; y < HEIGHT + 1; ++y) {
            if ((x + y + START_X + START_Y) % 2 == 1)
                continue
            var w = 1
            var h = 1
            xx = (x - 0.5)
            yy = (y - 0.5)
            if (x == 0) {
                xx += 0.5
                w = 0.5
            }
            if (y == 0) {
                yy += 0.5
                h = 0.5
            }
            if (x == WIDTH) {
                w = 0.5
            }
            if (y == HEIGHT) {
                h = 0.5
            }

            ctx.box(offx + xx * RES_PER_SQ, offy + yy * RES_PER_SQ, w * RES_PER_SQ, h * RES_PER_SQ)
            ctx.fill()
        }
    }

    //generate horizontal bits (2)
    for (var y = 0; y < HEIGHT; ++y) {
        for (var x = 0; x < WIDTH - 1; ++x) {
            p1 = [(1 + x) * RES_PER_SQ + offx, (0.5 + y) * RES_PER_SQ + offy]
            val = code2[(START_Y + y) * CODE_W + START_X + x]
            if ((val & 2) != 0) {
                ctx.setFillColor(1, 1, 1)
            }
            else {
                ctx.setFillColor(0, 0, 0)
            }
            ctx.circle(p1[0], p1[1], RES_PER_SQ / 6)
            ctx.fill()
        }
    }
    //generate vertical bits (1)
    for (var y = 0; y < HEIGHT - 1; ++y) {
        for (var x = 0; x < WIDTH; ++x) {
            p2 = [(0.5 + x) * RES_PER_SQ + offx, (1 + y) * RES_PER_SQ + offy]
            val = code2[(START_Y + y) * CODE_W + START_X + x]
            if ((val & 1) != 0)
                ctx.setFillColor(1, 1, 1)
            else
                ctx.setFillColor(0, 0, 0)
            ctx.circle(p2[0], p2[1], RES_PER_SQ / 6)
            ctx.fill()
        }
    }


    return ctx.writeAll()
}