from flask import Flask, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Interactive Perpendicular Line with Controls</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        #map { width: 100%; height: 600px; margin-bottom: 10px; }
        #controls { margin: 10px; font-size: 16px; }
        .latlon-display { margin-top: 5px; }
    </style>
</head>
<body>
    <h3>2点をクリック → 直交線表示 → スライダーで長さ調整 → マーカーをドラッグして再描画</h3>

    <div id="controls">
        📏 直交線の長さ（メートル）: 
        <input type="range" id="lengthSlider" min="10" max="200" value="50" step="1" />
        <span id="lengthValue">50</span> m

        <div class="latlon-display">
            📍端点1: <span id="point1">-</span><br>
            📍端点2: <span id="point2">-</span>
        </div>
    </div>

    <div id="map"></div>

    <script>
        // const map = L.map('map').setView([35.16910, 137.05414], 16);
        const map = L.map('map').setView([35.232, 138.888], 16);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        let markers = [];
        let baseLine = null;
        let perpLine = null;
        let perpLength = 50;

        const slider = document.getElementById("lengthSlider");
        const lengthValue = document.getElementById("lengthValue");
        const point1Display = document.getElementById("point1");
        const point2Display = document.getElementById("point2");

        slider.addEventListener("input", function() {
            perpLength = parseInt(this.value);
            lengthValue.textContent = perpLength;
            if (markers.length === 2) redrawLines();
        });

        function toFixed8(val) {
            return Number.parseFloat(val).toFixed(8);
        }

        function computePerpendicularLine(p1, p2, length_m) {
            const lat1 = p1[0], lon1 = p1[1];
            const lat2 = p2[0], lon2 = p2[1];
            const midLat = (lat1 + lat2) / 2;
            const midLon = (lon1 + lon2) / 2;

            const dx = lon2 - lon1;
            const dy = lat2 - lat1;

            const normal = [-dy, dx];
            const norm = Math.sqrt(normal[0]**2 + normal[1]**2);
            const unit = [normal[0]/norm, normal[1]/norm];

            const latMeter = 111000;
            const lonMeter = 111000 * Math.cos(midLat * Math.PI / 180);

            const dLat = (unit[1] * length_m / 2) / latMeter;
            const dLon = (unit[0] * length_m / 2) / lonMeter;

            const pt1 = [midLat + dLat, midLon + dLon];
            const pt2 = [midLat - dLat, midLon - dLon];

            return [pt1, pt2];
        }

        function redrawLines() {
            if (baseLine) map.removeLayer(baseLine);
            if (perpLine) map.removeLayer(perpLine);

            const p1 = markers[0].getLatLng();
            const p2 = markers[1].getLatLng();

            const point1 = [p1.lat, p1.lng];
            const point2 = [p2.lat, p2.lng];

            baseLine = L.polyline([point1, point2], { color: 'blue', weight: 4 }).addTo(map);

            const perpPts = computePerpendicularLine(point1, point2, perpLength);
            perpLine = L.polyline(perpPts, { color: 'red', weight: 2, dashArray: '5,5' }).addTo(map);

            point1Display.textContent = `(${toFixed8(perpPts[0][0])}, ${toFixed8(perpPts[0][1])})`;
            point2Display.textContent = `(${toFixed8(perpPts[1][0])}, ${toFixed8(perpPts[1][1])})`;
        }

        map.on('click', function(e) {
            if (markers.length >= 2) {
                markers.forEach(m => map.removeLayer(m));
                if (baseLine) map.removeLayer(baseLine);
                if (perpLine) map.removeLayer(perpLine);
                markers = [];
                point1Display.textContent = "-";
                point2Display.textContent = "-";
            }

            const lat = e.latlng.lat;
            const lon = e.latlng.lng;

            const marker = L.marker([lat, lon], { draggable: true }).addTo(map)
                .bindPopup('Lat: ' + toFixed8(lat) + '<br>Lon: ' + toFixed8(lon))
                .openPopup();

            marker.on('dragend', function() {
                if (markers.length === 2) redrawLines();
            });

            markers.push(marker);

            if (markers.length === 2) redrawLines();
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True)
