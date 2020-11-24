var VG = (function(vg) {

    vg.drawPoint= function(ctx, point, wStart, hStart, scale, pointSize) {
        ctx.fillStyle = "#ff2626"; // Red color
        //var pointSize = 3.5;
        
        x = (point[1]+wStart)*scale;
        y = (point[0]+hStart)*scale;
        //console.log(point[1], point[0]);
        //console.log(x,y);

        ctx.beginPath();
        ctx.arc(x, y, pointSize, 0, Math.PI * 2, true);
        ctx.fill();

        // Fill in a ring
        ctx.beginPath();
        ctx.strokeStyle = '#42f44b';
        var circleSize = pointSize*2;
        ctx.arc(x, y, circleSize, 0, 2* Math.PI);
        ctx.stroke()

        //Surround in a ring
        //ctx.strokeStyle = '#42f44b';
        ctx.beginPath();
        ctx.strokeStyle =  "#ff2626"; 
        var circleSize = pointSize*3;
        ctx.arc(x, y, circleSize, 0, 2* Math.PI);
        ctx.stroke()
        //console.log("Drew point (" + point + ") ");
    }
    
    return vg;
    
}(VG || {}));
