# Startlens(画像データの学習)


## 概要

Startlensのwebアプリケーション（参照にGitHubリンク）の観光事業者側のサイトから投稿した画像をDeep Metric Learningにより距離学習を行い類似画像検索機能をAPI経由で提供する。
今後は、モバイル端末からAPIリクエストで登録した画像の検索が行えるようにする。


## 機能

- Deep metric learning（triplet Network）による画像データのベクトル変換処理
　 - 観光事業者ごとにS3 storageから画像データをオンメモリで取得・加工処理し学習を行う
   - Hdf5形式のファイルにより学習パラメータの保存とS3へのバックアップ機能
   
- KNN法によるベクトルデータの学習
　 - Tripelet Networkで変換したベクトルデータから距離が近いデータを類似画像とみなし同じクラスに分類する学習をする
   - 新しい画像が更新されるたびにTriplet Networkによる変換処理が実行されるとサーバーに負荷がかかることから、キャッシュとして一度変換した画像データをcsv形式で保存・S3へのバックアップ処理を実行する

- KNN法による推論処理
　 - 学習済みのHdf5ファイルをtflite形式に変換し、Deep Learningの推論処理（画像のベクトル変換）はモバイル等のエッジ端末で行う想定
   - API経由で取得したベクトルデータをKNN法により類似画像分類を行う
   
- 訓練画像データの追加登録
　 - 全てのデータを再度一斉に学習しないように追加した画像（特定のクラスラベル）のみ学習できるように実装
  
- FlaskによるAPIエンドポイントの実装

- pytestによるユニットテスト


## 開発環境

python: 3.7.9

tensorflow: 2.4.1

scikit-learn: 0.24.1


## 環境構築

- Dokerによる開発環境構築

```
$ git clone https://github.com/yuta252/startlens_learning.git

$ docker-compose up --build -d
```

- アプリケーションの起動方法

   - TripletNetworkによる学習
   ```
   $ python main.py -m train
   ```
   
   - APIサーバーの起動
   ```
   $ python main.py -m server
   ```


## 参照

- Startlens Rails backend API (https://github.com/yuta252/startlens_web_backend)
- Startlens frontend(ユーザー側）（https://github.com/yuta252/startlens_frontend_user）
- Startlens frontend(観光事業者側)（https://github.com/yuta252/startlens_react_frontend）
