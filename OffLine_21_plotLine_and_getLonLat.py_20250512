import folium
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Flask用のHTMLテンプレート
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Click to Draw Line</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
</head>
<body>
    <h3>クリックして2点を選択し、ラインを引く</h3>
    <div id="map" style="width: 100%; height: 600px;"></div>
    
    <script>
        // var map = L.map('map').setView([35.232, 138.888], 16);
        var map = L.map('map').setView([35.16910, 137.05414], 16);
        
        // タイルレイヤー追加
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        var click_points = [];

        map.on('click', function(e) {
            var lat = e.latlng.lat;
            var lon = e.latlng.lng;
            click_points.push([lat, lon]);

            // クリックした位置にマーカーを追加
            L.marker([lat, lon]).addTo(map)
                .bindPopup('Lat: ' + lat.toFixed(6) + '<br>Lon: ' + lon.toFixed(6))
                .openPopup();

            if (click_points.length === 2) {
                // ラインを描画
                L.polyline(click_points, {color: 'blue', weight: 4}).addTo(map);

                // Python にデータを送信（Flask経由）
                fetch('/print_points', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(click_points)
                });

                // リセット
                click_points = [];
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/print_points", methods=["POST"])
def print_points():
    points = request.json
    point1 = tuple(points[0])  # (lat1, lon1)
    point2 = tuple(points[1])  # (lat2, lon2)
    
    formatted_output = f"(({point1[0]}, {point1[1]}), ({point2[0]}, {point2[1]}), 0)"
    print(formatted_output)  # ターミナルに出力

    return "", 200

if __name__ == "__main__":
    app.run(debug=True)
