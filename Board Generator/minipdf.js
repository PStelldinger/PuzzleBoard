
function pad(num, size) {
    var s = "000000000000" + num;
    return s.substr(s.length - size);
}
class PDFCanvas {
    constructor(title = "page.pdf", width = 100, height = 200, originTopLeft = true) {
        this.title = title;
        this.canvas = "";
        if (originTopLeft) {
            this.setMatrix(1, 0, 0, -1, 0, height);
        }
        this.outbuf = ""
        this.canvasend = `endstream
`;
        this.objs = [
            `<<
/Type /Catalog
/Pages 3 0 R
>>
`,
            `<<
/Type /Pages
/Kids [4 0 R ]
/Count 1
>>
`,
            `<< 
/Type /Page
/Parent 3 0 R
/MediaBox [0 0 ${width} ${height}] 
/Contents 1 0 R
>>
`
        ];
        this.writeCount = 0;
    }
    write(obj) {
        this.writeCount += obj.length;
        //console.log(obj);
        this.outbuf += obj
    }

    writeAll() {
        this.outbuf = ""
        this.writeCount = 0;
        this.write("%PDF-1.4\n");
        var offsets = [];

        //first object is our canvas
        offsets.push(this.writeCount);
        this.write("1 0 obj\n");
        this.write(
            `<<
/Length ${this.canvas.length}
>>
stream
`);
        this.write(this.canvas);
        this.write(this.canvasend);
        this.write("endobj\n");
        for (var k = 0; k < this.objs.length; ++k) {
            let obj = this.objs[k];
            let i = k + 2;
            offsets.push(this.writeCount);
            this.write(`${i} 0 obj\n`);
            this.write(obj);
            this.write("endobj\n");
        }
        let xrefpos = this.writeCount;
        this.write(`\n\nxref\n0 ${this.objs.length}\n0000000000 65535 f\n`);
        for (var k = 0; k < offsets.length; ++k) {
            let o = offsets[k];
            let i = k + 1;
            this.write(`${pad(o + i, 10)} 00000 n\n`);

            this.write(
                `trailer
<<
/Size ${this.objs.length}
/Info
<<
/Producer(nos_minipdf)
/Title(${this.title})
>>
/Root 2 0 R
>>
startxref
${xrefpos}
%% EOF`);
            return this.outbuf;
        }
    }

    setFillColor(r, g, b) {
        this.canvas += `${r} ${g} ${b} rg\n`;
    }
    setStrokeColor(r, g, b) {
        this.canvas += `${r} ${g} ${b} RG\n`;
    }
    setStrokeWidth(width) {
        this.canvas += `${width} w\n`;
    }
    stroke() {
        this.canvas += "s\n";
    }
    fill() {
        this.canvas += "f\n";
    }
    strokeAndFill() {
        this.canvas += "fs\n";
    }
    setMatrix(xx, xy, yx, yy, ox, oy) {
        this.canvas += `${xx} ${xy} ${yx} ${yy} ${ox} ${oy} cm\n`;
    }
    box(x, y, w, h) {
        //console.log("box", x, y, w, h)
        this.canvas += `${x} ${y} m ${x} ${y + h} l ${x + w} ${y + h} l ${x + w} ${y} l h\n`;
    }
    beginPath() {
        this.canvas += "q\n"
    }
    endPath() {
        this.canvas += "Q\n"
    }
    circle(centerx, centery, radius) {
        this.canvas += `% %% circle %%
% startpoint
${centerx + radius} ${centery} m
% bezier control_point1 control_point2 dst
% 1st quarter bezier
${centerx + radius} ${centery + 0.5523 * radius} ${centerx + 0.5523 * radius} ${centery + radius} ${centerx} ${centery + radius} c
% 2nd quarter bezier
${centerx - 0.5523 * radius} ${centery + radius} ${centerx - radius} ${centery + 0.5523 * radius} ${centerx - radius} ${centery} c
% 3rd quarter bezier
${centerx - radius} ${centery - 0.5523 * radius} ${centerx - 0.5523 * radius} ${centery - radius} ${centerx} ${centery - radius} c
% 4th quarter bezier
${centerx + 0.5523 * radius} ${centery - radius} ${centerx + radius} ${centery - 0.5523 * radius} ${centerx + radius} ${centery} c
`
    }
    text(text, x, y) {
        this.canvas += `BT
% font size ?
/F1 72 Tf
1 0 0 1 ${x} ${y} Tm
48 TL
(${text})'
ET
`
    }
}