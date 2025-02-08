from flask import Flask, render_template_string
import folium

app = Flask(__name__)

@app.route('/')
def index():
    # 初期位置を指定
    # start_coords = (35.0169, 137.2908) # 下山第3周回路
    # start_coords = (34.9158, 134.2189) # 岡山国際サーキット
    # start_coords = (38.1422, 140.7778) # SUGOサーキット
    start_coords = (35.0549, 137.1631) # 本社T/C
    folium_map = folium.Map(location=start_coords, zoom_start=18) # ズームレベルを上げました

    # ポップアップ機能を追加
    folium.LatLngPopup().add_to(folium_map)

    # 地図をHTML形式に変換
    map_html = folium_map._repr_html_()

    # デバッグ出力
    print('Debug: Generating map HTML...')
    print('map_html:', map_html[:500])  # 最初の500文字を出力

    # HTMLテンプレートとしてレンダリング
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Map</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                #map { position:absolute; top:0; bottom:0; right:0; left:0; }
            </style>
        </head>
        <body>
            <div id="map">{{ map_html|safe }}</div>
        </body>
        </html>
    ''', map_html=map_html)

if __name__ == '__main__':
    print('Starting Flask server...')
    app.run(debug=True)
