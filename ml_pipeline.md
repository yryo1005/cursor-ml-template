@my_utils.py内の関数は事前に私が定義した機能です．
適時my_utils.pyのコードを参照にしてください．
概要は以下の通りです．

### set_seed関数
- 概要
    実験の再現性を担保するための関数
- 引数
    - seed (int) = 0 
- 戻り値
    なし
- 処理
    - Python，Numpy，Pytorch，CUDAのseed値を固定する．
    - CUDA等の処理を決定論的になるよう設定する．

### ResultLoggerクラス
- 概要
    実験中の各種メトリクス（Loss，Accuracyなど）の推移をメモリ上に記録し，JSONファイルとして保存・読み込みを行うための軽量なロガークラス．

- 初期化 (\_\_init\_\_)
    - 引数
        target_path (str) = None
    - 処理
        - 記録する指標名を管理する names と，履歴データを保持する辞書 history を初期化する．
        - target_path が指定された場合は，自動的に過去のログファイルを読み込む（load メソッドを実行する）．

- set_names メソッド
    - 概要
        記録したい指標の名前を登録する。
    - 引数
        *names (strの可変長引数) （例: "loss", "accuracy"）
    - 例外
        すでに names が登録されている場合は例外（エラー）を発生させる．
    - 処理
        - 引数で受け取った名前のリストを作成し，history 辞書にそれぞれの空のリストを用意する．すでに存在する指標名がある場合は初期化をスキップする．

- \_\_call\_\_ メソッド
    - 概要
        インスタンスを関数のように呼び出し，指標の値を履歴に追加（append）する．
    - 引数 
        *values (float/intの可変長引数)
    - 例外
        set_names が未実行の場合，または渡された値の数が登録された指標の数と一致しない場合は例外を発生させる．
    - 処理
        登録されている指標名と，渡された値を順番にペア（zip）にし，対応する履歴リストの末尾に値を追加する．

- save メソッド
    - 概要
        記録された履歴（history）をJSONファイルとして保存する．
    - 引数
        target_path (str)

- load メソッド
    - 概要
        保存されたJSONファイルから履歴データを読み込む．
    - 引数
        target_path (str)
    - 処理
        JSONファイルから辞書データを読み込み，history に格納する．また，辞書のkeyのリストを names に自動設定する。

- \_\_getitem\_\_ メソッド
    -概要
        辞書のようにブラケット [] を使って，特定の指標の履歴リストを取得する．
    - 引数
        key (str)
    - 戻り値
        list （指定した指標の履歴リスト。存在しない場合は空のリスト [] を返す）

プログラムは以下の思想に則り記述してください．

### load_dataloader関数
- 概要
    学習/検証用のデータローダーをインスタンス化するための関数
- 引数
    - seed (int) = 0
- 戻り値
    - train_dataloader (torch.utils.Dataloader)
    - test_dataloader (torch.utils.Dataloader)
- 処理
    - 独自のデータセットの場合，sklearn.random_stateをseedに指定してmodel_selection.train_test_splitを用いて学習用と検証用のデータセットに分割する
    - seedを引数としてset_seed関数を実行し，データの並びの初期値を固定する
    - seedを指定したtorch.Generatorを用いてデータのシャッフルを決定論的にする

### load_model関数
- 概要
    モデルをインスタンス化するための関数
- 引数
    - ModelClass (torch.nn.Moduleのクラス)
    - weight_path (str) = None
    - seed (int) = 0
- 戻り値
    model (torch.nn.Module)
- 処理
    - seedを引数としてset_seed関数を実行し，パラメータの初期値を固定する
    - ModelClassをインスタンス化する
    - weight_pathがNoneでない場合，weight_pathのパラメータをモデルに読み込む

### loss_func関数
- 概要
    モデルの出力と教師信号の誤差を計算する関数
- 引数
    - outputs (torch.Tensor); モデルの出力
    - teacher_signals (torch.Tensor); 教師信号
- 戻り値
    loss (torch.Tensor)
- 処理
    - outputsとteacher_signalsの誤差lossを計算する
    - 誤差関数はタスクに応じて定める
    - 誤差の総和をデータ数で割った，1データあたりの誤差の平均を計算する

### metrics_func関数
- 概要
    モデルの出力と教師信号の一致度を評価する関数
- 引数
    - outputs (torch.Tensor); モデルの出力
    - teacher_signals (torch.Tensor); 教師信号
- 戻り値
    metrics_to_value (dict); 評価指標の名前がkey，値がvalueの辞書
- 処理
    - outputsとteacher_signalsの一致度を評価する
    - 評価(Accuracy, Perplexityなど)はタスクに応じて定める
    - 評価の総和をデータ数で割った，1データあたりの誤差の平均を計算する
    - 誤差はiteration関数内で計算するため，この関数内では計算しない

### iteration関数
- 概要
    1つのミニバッチのデータを学習/検証する関数
- 引数
    - model (torch.nn.Module); モデル
    - inputs (torch.Tensor); 入力
    - teacher_signals (torch.Tensor);  教師信号
    - optimizer (torch.optim.Optimizer) = None; 最適化手法
- 戻り値
    - metrics_to_value (dict); 評価指標の名前がkey，値がvalueの辞書
- 処理 
    - モデルの出力 (outputs) を計算する
    - 誤差を計算する
    - optimizerがNoneではない場合，勾配を計算し，最適化手法でパラメータを更新する
    - 評価を計算する
    - 誤差と評価がkey，値がvalueの辞書を作成する

### epoch関数
- 概要
    1つのデータローダーの全データを学習/検証する関数
- 引数
    - model (torch.nn.Module); モデル
    - dataloader (torch.utils.Dataloader); データローダー
    - optimizer (torch.optim.Optimizer) = None; 最適化手法
- 戻り値
    - metrics_to_value (dict); 評価指標の名前がkey，値がvalueの辞書
- 処理
    - この関数の中でiteration関数を定義する
    - dataloaderを繰り返し条件，(inputs, teacher_signals)を繰り返し変数とするループを回す
    - modelのデバイスを確認し，inputsとteacher_signalsを同じデバイスに移動させる
    - model, inputs, teacher_signals, optimizerを引数としてiteration関数を実行する
    - iteration関数の戻り値は平均値であるため，ミニバッチのデータ数を掛け総和に戻す
    - tqdmの進捗バーにその反復時点の1データあたりの評価の平均値を出力する
    - 全ミニバッチにおける評価の合計を全データ数で割り1データあたりの評価の平均値を計算する

### train関数
-　概要
    モデルを学習する関数
- 引数
    - target_dir (str); モデルの重みや実験結果を保存するディレクトリ
    - ModelClass (torch.nn.Moduleのクラス)
    - load_dataloader (func)
    - epochs (int)
    - batch_size (int)
    - seed (int) = 0
- 戻り値
    なし
- 処理
    - この関数の中でepoch関数を定義する
    - seed, batch_sizeを引数としてload_dataloader関数を実行する
    - seedを引数としてload_model関数を実行する
    - GPUが使用できるか確認し，使用できる場合modelをGPUに移動させる
    - 学習データに対する評価，検証データに対する評価を引数としてResultLoggerを変数名をloggerとしてインスタンス化する
    - epochsの回数だけepoch関数を実行する
    - epochが終了するたびに，学習データと検証データに対するmetrics_to_valueをlogger.__call__の引数に与える
    - epochが終了するたびに，特定の評価が全ての反復において最も優れているか確認し，最も優れている場合modelの重みを保存する
    - 0epoch目は学習せず，初期値での結果を取得する

### 設計思想・結果管理ルール
- 学習結果の保存
    - 学習結果は，以下の規則でディレクトリを作成し，保存してください
      `outputs/{data_name}/{model_name}/{ハイパーパラメータ}/{seed}`
      ex) `outputs/mnist/cnn/0.001_32/0`
    - すでに結果がある条件で学習を開始した場合，それをスキップしてください
    
- 学習の実行
    - Modelクラスを記述したmodel.pyやload_dataloader関数等をまとめたutils.pyは，以下の規則でディレクトリを作成し，保存してください
      `{data_name}/{model_name}`
      ex) `mnist/cnn`
    - 学習はプログラム終了後もログを残すために，train.ipynb内で実行してください
    - ハイパーパラメータの探索空間やseed数はtrain.ipynb内で指定してください
    - seed数だけ全てのハイパーパラメータの組みで学習してください

- 学習結果の可視化
    - ルートディレクトリに実験結果を可視化するvisualaize_result.ipynbを実装してください
    - 学習結果を保存しているoutputsディレクトリ内を再帰的に捜査し，同条件における全seedの結果から平均，標準偏差を計算し信頼区間をプロットしてください．