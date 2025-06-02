from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Leaflet Map Click</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
            <style>
                #map { position:absolute; top:0; bottom:0; right:0; left:0; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
            <script>
                var map = L.map('map').setView([35.22703636383409, 138.90208082942948], 18);

                // OpenStreetMap のタイルレイヤーを追加
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 19,
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);

                // クリック時の処理
                map.on('click', function(e) {
                    var lat = e.latlng.lat.toFixed(8);
                    var lng = e.latlng.lng.toFixed(8);
                    L.popup()
                        .setLatLng(e.latlng)
                        .setContent("Lat: " + lat + "<br>Lng: " + lng)
                        .openOn(map);
                });
            </script>
        </body>
        </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)
