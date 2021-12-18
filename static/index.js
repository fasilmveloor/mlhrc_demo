let canvas;
window.onload = function() {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.freeDrawingBrush.width = 8;
    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush.color = '#000000';
    canvas.renderAll();
    const pause_btn = document.getElementById('freedraw');
    pause_btn.addEventListener('click', function() {
        canvas.isDrawingMode = !canvas.isDrawingMode;
        if (canvas.isDrawingMode) {
            pause_btn.innerHTML = "Pause freedraw";
        } else {
            pause_btn.innerHTML = "Enable freedraw";
        }
    });
};
let submitImage = function() {
    document.getElementById("loading").classList.add("fa", "fa-spinner", "fa-spin", "fa-3x", "fa-fw");
    document.getElementById("result").innerText = '';
    axios.post('/predict', {
        img: canvas.toDataURL('png')
    }).then(function(response) {
        document.getElementById("loading").className = "";
        let data = response.data;
        console.log(data);
        const ch = data;
        document.getElementById("result").innerText = ch;
        // if(data.length>5)
        //      data = data.slice(0,5);
        //drawGraph(data);
    });
};
let clearCanvas = function() {
    canvas.clear();
};

function drawGraph(data) {
    let i = 1;
    for (let d in data) {
        const e = data[d];
        console.log(e);
        if (e[1] * 100 > 1) {
            document.getElementById('label' + i).innerText = String.fromCharCode(e[0]);
            document.getElementById('graph' + i).style.height = e[1] * 100 + 'px';
        }
        i++;
    }
}