# ThreeDPoseUnitySample

画像、動画、カメラなどの２次元の画像データから人体の３次元の姿勢を推定する機械学習を研究しています。こちらの内容は[私のTwitterアカウント](https://twitter.com/yukihiko_a)を参照ください。

ThreeDPoseUnitySampleは、その学習結果のモデルとUnityを使用した実装サンプルです。姿勢推定は一人が写っている画像を前提としています。複数人の推定には対応していません。

The next version uses Unity Barracuda. It has been published in this repository.

 <img src='https://github.com/yukihiko/ThreeDPoseUnitySample/blob/master/samplemv.gif' width=600/>

## 使い方
Video Playerに動画をセットします。この動画からclipRectのサイズに切り出し、TextureObjectを介して姿勢推定用のonnxに渡されます。
わざわざTextureObjectに渡す必要はありませんが、入力画像が224ｘ224を想定しているためその為の確認用です。

動画とclipRectを正しく設定すれば動くはずですが、姿勢推定のモデルはまだ研究中ですのでそれほどの精度がありません。それなりの精度を出すためには、
- 背景がシンプルであること（床の反射とかも誤認識する事があります）
- 常に全身が写っていること（全身が写っている事が前提で作っています）
- 人物が大きくもなく小さくもなく
- ダボっとした服は誤認識しやすいです。手足がわかる服が良いです


サンプルに使用している動画「wiper.mp4」は[ミソジサラリーマン様](https://www.youtube.com/user/tanahiro814)の[こちらの動画](https://www.youtube.com/watch?v=C9VtSRiEM7s)を使用させて頂いております。ありがとうございます。このファイルは許可なく動画サイト等への転載は行わないでください。


# License
非営利目的の使用のみ可です。趣味・研究などにはご自由にお使いください。再配布する場合はクレジット（Digital-Standard Co.,Ltd.）を入れていただけるとありがたいです。営利目的で使用したい場合はご相談ください。

Non-commercial use only.Please use it freely for hobbies and research. When redistributing, it would be appreciated if you could enter a credit (Digital-Standard Co., Ltd.).Please contact us if you want to use it for commercial purposes.

