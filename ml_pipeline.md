# 共通前提指示
あなたは機械学習の高度なエンジニアです。
提供された設計思想、データ構造、および関数仕様を完全に理解し、これらに厳格に従ってプログラムの作成や環境構築を行ってください。
数式を記述する場合は、必ずTeX表記（$ inline $ または $$display$$）を使用してください。
また、日本語の文章における句読点には、必ずカンマ「，」とピリオド「．」を使用してください。

---

# 1. 外部モジュール仕様 (@my_utils.py)
※以下の関数およびクラスは、事前に `my_utils.py` に定義されています。適宜参照・利用してください。

### set_seed 関数
- **概要**: 実験の再現性を担保するための関数．
- **引数**: `seed (int) = 0`
- **戻り値**: なし
- **処理**: Python，NumPy，PyTorch，CUDAのseed値を固定する．CUDA等の処理を決定論的になるよう設定する．

### ResultLogger クラス
- **概要**: 実験中の各種メトリクス（Loss，Accuracyなど）の推移をメモリ上に記録し，JSONファイルとして保存・読み込みを行うための軽量なロガークラス．
- **初期化 (__init__)**:
  - 引数: `target_path (str) = None`
  - 処理: 記録する指標名を管理する `names` と，履歴データを保持する辞書 `history` を初期化する．`target_path` が指定された場合は，自動的に過去のログファイルを読み込む（`load` メソッドを実行）．
- **set_names メソッド**:
  - 概要: 記録したい指標の名前を登録する．
  - 引数: `*names (str)` （例: "loss", "accuracy"）
  - 例外: すでに `names` が登録されている場合は例外を発生させる．
  - 処理: 引数で受け取った名前のリストを作成し，`history` 辞書にそれぞれの空のリストを用意する．すでに存在する指標名がある場合は初期化をスキップする．
- **__call__ メソッド**:
  - 概要: インスタンスを関数のように呼び出し，指標の値を履歴に追加（append）する．
  - 引数: `*values (float/int)`
  - 例外: `set_names` が未実行の場合，または渡された値の数が登録された指標の数と一致しない場合は例外を発生させる．
  - 処理: 登録されている指標名と，渡された値を順番にペア（`zip`）にし，対応する履歴リストの末尾に値を追加する．
- **save メソッド**:
  - 概要: 記録された履歴（`history`）をJSONファイルとして保存する．
  - 引数: `target_path (str)`
- **load メソッド**:
  - 概要: 保存されたJSONファイルから履歴データを読み込む．
  - 引数: `target_path (str)`
  - 処理: JSONファイルから辞書データを読み込み，`history` に格納する．また，辞書のkeyのリストを `names` に自動設定する．
- **__getitem__ メソッド**:
  - 概要: 辞書のようにブラケット `[]` を使って，特定の指標の履歴リストを取得する．
  - 引数: `key (str)`
  - 戻り値: `list` （指定した指標の履歴リスト．存在しない場合は空のリスト `[]` を返す）

---

# 2. 実装すべき共通関数・クラス仕様
※作成する各関数には、概要、引数のデータ型/形状、戻り値のデータ型/形状をコメント（Docstring）で明記すること．

### load_dataloader 関数
- **概要**: 学習/検証用のデータローダーをインスタンス化するための関数．
- **引数**: `seed (int) = 0`
- **戻り値**: `train_dataloader (torch.utils.data.DataLoader)`, `test_dataloader (torch.utils.data.DataLoader)`
- **処理**: 
  - 独自のデータセットの場合，`sklearn.model_selection.train_test_split` を用い，`random_state` に `seed` を指定して学習用と検証用のデータセットに分割する．
  - `seed` を引数として `set_seed` 関数を実行し，データの並びの初期値を固定する．
  - `seed` を指定した `torch.Generator` を用いてデータのシャッフルを決定論的にする．

### load_model 関数
- **概要**: モデルをインスタンス化するための関数．
- **引数**: `ModelClass (torch.nn.Moduleのクラス)`, `weight_path (str) = None`, `seed (int) = 0`
- **戻り値**: `model (torch.nn.Module)`
- **処理**: 
  - `seed` を引数として `set_seed` 関数を実行し，パラメータの初期値を固定する．
  - `ModelClass` をインスタンス化する．
  - `weight_path` が `None` でない場合，`weight_path` のパラメータをモデルに読み込む．

### loss_func 関数
- **概要**: モデルの出力と教師信号の誤差を計算する関数．
- **引数**: `outputs (torch.Tensor)`（モデルの出力）, `teacher_signals (torch.Tensor)`（教師信号）
- **戻り値**: `loss (torch.Tensor)`
- **処理**: 
  - `outputs` と `teacher_signals` の誤差 `loss` を計算する（誤差関数はタスクに応じて定める）．
  - 誤差の総和をデータ数で割った，1データあたりの誤差の平均を計算する．

### metrics_func 関数
- **概要**: モデルの出力と教師信号の一致度を評価する関数．
- **引数**: `outputs (torch.Tensor)`（モデルの出力）, `teacher_signals (torch.Tensor)`（教師信号）
- **戻り値**: `metrics_to_value (dict)` （評価指標の名前がkey，値がvalueの辞書）
- **処理**: 
  - `outputs` と `teacher_signals` の一致度を評価する（評価指標は Accuracy, Perplexity などタスクに応じて定める）．
  - 評価の総和をデータ数で割った，1データあたりの評価の平均を計算する．
  - ※誤差（Loss）は `iteration` 関数内で計算するため，この関数内では計算しない．

### iteration 関数
- **概要**: 1つのミニバッチのデータを学習/検証する関数．
- **引数**: `model (torch.nn.Module)`, `inputs (torch.Tensor)`, `teacher_signals (torch.Tensor)`, `optimizer (torch.optim.Optimizer) = None`
- **戻り値**: `metrics_to_value (dict)` （誤差と評価指標の名前がkey，値がvalueの辞書）
- **処理**: 
  - モデルの出力（`outputs`）を計算する．
  - `loss_func` を用いて誤差を計算する．
  - `optimizer` が `None` ではない場合，勾配を計算し，最適化手法でパラメータを更新する．
  - `metrics_func` を用いて評価を計算する．
  - 誤差と評価がkey，値がvalueの辞書を作成して返す． 

### epoch 関数
- **概要**: 1つのデータローダーの全データを学習/検証する関数．
- **引数**: `model (torch.nn.Module)`, `dataloader (torch.utils.data.DataLoader)`, `optimizer (torch.optim.Optimizer) = None`
- **戻り値**: `metrics_to_value (dict)` （全データにおける1データあたりの評価・誤差の平均値の辞書）
- **処理**: 
  - この関数の中で `iteration` 関数を定義（または呼び出し）する．
  - `dataloader` を繰り返し条件，`(inputs, teacher_signals)` を繰り返し変数とするループを回す．
  - `model` のデバイスを確認し，`inputs` と `teacher_signals` を同じデバイスに移動させる．
  - `iteration` 関数を実行する．戻り値はミニバッチ内の平均値であるため，ミニバッチのデータ数を掛け合わせて総和に戻し，エポック全体の合計に累積する．
  - `tqdm` の進捗バーに，その反復（iteration）時点における1データあたりの評価・誤差の平均値を出力する．
  - 全ミニバッチ終了後，評価・誤差の合計を全データ数で割り，1データあたりの平均値を計算して返す．

### train 関数
- **概要**: モデルを学習する関数．
- **引数**: `target_dir (str)`（結果保存先ディレクトリ）, `ModelClass (torch.nn.Moduleのクラス)`, `load_dataloader (func)`, `epochs (int)`, `batch_size (int)`, `seed (int) = 0`
- **戻り値**: なし
- **処理**: 
  - この関数の中で `epoch` 関数を定義（または呼び出し）する．
  - `seed`, `batch_size` を引数として `load_dataloader` 関数を実行する．
  - `seed` を引数として `load_model` 関数を実行する．
  - GPUが使用できるか確認し，使用できる場合は `model` をGPUに移動させる．
  - `ResultLogger` を変数名 `logger` としてインスタンス化する（学習・検証データの各種メトリクス名を設定）．
  - **0エポック目（初期状態）**: 学習は行わず，初期値での検証結果を取得して `logger` に記録する．
  - `epochs` の回数だけ `epoch` 関数による学習および検証を実行する．
  - 各エポック終了後，それぞれの結果を `logger.__call__` に与えて記録する．
  - 各エポック終了後，特定の評価（例: Validation Accuracy）が過去すべての反復において最も優れているか確認し，最高値を更新した場合は `model` の重みを保存する．

---

# 3. 設計思想・結果管理ルール

### 環境構築
- Pythonの仮想環境は `venv` で管理する．
- 使用するライブラリとそのバージョンを `requirements.txt` に記述し，これに基づきインストールする．
- 学習結果を保存するディレクトリ（`outputs/`）やデータセットのディレクトリは，サイズが巨大になるため `.gitignore` に追加する．

### コードのモジュール化と実験ディレクトリ
- モデルクラスを記述した `model.py` や，`load_dataloader` 関数等をまとめた `utils.py` は，実験ごとに以下の規則でディレクトリを作成して保存すること．
  `ex_{n:03}_{全ての比較手法で共通の条件}/` （※ `n` は実験の通し番号）
  - 例1: `ex001_mnist_cnn/`
  - 例2: `ex002_cifar10_resnet/`

### 学習の実行
- プログラム終了後もログ（実行結果や出力）を残すために，学習は必ず各実験ディレクトリ内の `train.ipynb`（Jupyter Notebook）で実行すること．
- ハイパーパラメータの探索空間や `seed` 数は `train.ipynb` 内で指定する．
- 設定された `seed` 数の分だけ，すべてのハイパーパラメータの組み合わせについてループを回して学習を実行する．
- **重複実行の回避**: すでに結果（ログファイルや重み）が存在する条件のフォルダがある場合は，その条件の学習をスキップ（パス）すること．

### 学習結果の保存規則
- 実験結果は以下のディレクトリ階層に厳格に従って保存すること．
  `outputs/ex_{n:03}_{全ての比較手法で共通の条件}/{比較する手法}/{ハイパーパラメータ}/{seed}/`
  - 例: `outputs/ex001_mnist_cnn/Adam/0.001_32/0/`

### 学習結果の可視化 (`visualize_result.ipynb`)
- プロジェクトのルートディレクトリに，実験結果を可視化するための `visualize_result.ipynb` を実装すること．
- `outputs/` ディレクトリ内を再帰的に走査し，同一条件における全 `seed` の結果から平均と標準偏差を計算し，信頼区間（エラーバーや塗りつぶし）を含めたプロットを行うこと．
- 可視化したい「比較手法」や「ハイパーパラメータ」を，プログラム上の変数やリストで指定・選択できるように設計すること．
- `matplotlib` でグラフを作成する際は，原則として正方形（例: `figsize=(5, 5)`）にすること．ただし，文字やラベルがはみ出るなどの問題がある場合は，視認性を最優先して適切なサイズに調整すること．

### 仕様書・レポートの作成
- プログラムの作成・整備が終了した段階で，メインロジックや関数の関係性を説明した `document.md` を作成すること．
- すべての実験が終了した後に，これまでに実施した実験結果（グラフや考察）をまとめた `report.md` を作成すること．

# ディレクトリ構造の例
```text
.
├── .gitignore               # outputs/ や datasets/、.venv/ を除外
├── requirements.txt         # 依存ライブラリとバージョンを固定
├── my_utils.py              # 事前に定義された set_seed, ResultLogger が含まれる共通モジュール
├── visualize_result.ipynb   # outputs/ 内を走査して信頼区間付きグラフを描画するノートブック
│
├── ex001_mnist_cnn/         # 実験001: 全比較手法で共通の条件（例: MNISTに対するCNN）
│   ├── model.py             # この実験で使用するモデル構造（CNN）の定義
│   ├── utils.py             # load_dataloader, load_model, 各種関数を定義
│   └── train.ipynb          # ハイパーパラメータやseedのループを回して学習を実行する主体
│
├── ex002_cifar10/           # 実験002: （例: CIFAR-10に対する実験）
│   ├── model.py
│   ├── utils.py
│   └── train.ipynb
│
├── outputs/                 # 実験結果の出力先（.gitignoreに対象設定）
│   ├── ex001_mnist_cnn/
│   │   ├── SGD/             # 比較する手法 1
│   │   │   ├── 0.001_32/    # ハイパーパラメータ（学習率_バッチサイズ）
│   │   │   │   ├── 0/       # seed 0 の結果
│   │   │   │   │   ├── best_model.pth
│   │   │   │   │   └── log.json
│   │   │   │   └── 1/       # seed 1 の結果
│   │   │   │       ├── best_model.pth
│   │   │   │       └── log.json
│   │   │   └── 0.01_64/
│   │   │       └── ...
│   │   └── Adam/            # 比較する手法 2
│   │       └── ...
│   └── ex002_cifar10/
│       └── ...
│
├── document.md              # メインロジックやコード仕様を解説したドキュメント
└── report.md                # 実施した全実験の結果や考察をまとめたレポート
```
