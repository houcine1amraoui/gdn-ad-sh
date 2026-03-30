import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.downsample import detect_sensor_types
from src.utils.load_actors_timelines import load_actor_timelines

def filter_columns_one_data(df, config):

    
    project_root_dir = config["project_root_dir"]
    dataset_name = config["dataset"]["dataset_name"]
    dataset_folder = config["dataset"]["dataset_folder"]
    filtered_columns_path = f"{project_root_dir}/{dataset_folder}/{dataset_name}_filtered_columns.txt"

    print(f"Filtering {dataset_name} data...")

    # read column names from txt
    with open(filtered_columns_path, "r") as f:
        filtered_columns = [line.strip() for line in f if line.strip()]

    # keep only existing columns
    selected_columns = [c for c in filtered_columns if c in df.columns]

    # filter dataframe
    df_filtered = df[selected_columns]
    return df_filtered

def filter_columns_merged_data(df, config):
    print("Filtering BRE+CU merged data...")
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]

    bre_filtered_columns_path = f"{project_root_dir}/{dataset_folder}/BRE_filtered_columns.txt"
    cu_filtered_columns_path = f"{project_root_dir}/{dataset_folder}/CU_filtered_columns.txt"

    # read BRE columns
    with open(bre_filtered_columns_path, "r") as f:
        bre_filtered_columns = [line.strip() for line in f if line.strip()]

    # read CU columns
    with open(cu_filtered_columns_path, "r") as f:
        cu_filtered_columns = [line.strip() for line in f if line.strip()]

    # merge column lists correctly
    all_filtered_columns = bre_filtered_columns + cu_filtered_columns

    # remove duplicates
    all_filtered_columns = list(set(all_filtered_columns))

    # keep timestamp
    if "Timestamp" not in all_filtered_columns:
        all_filtered_columns.append("Timestamp")

    # keep only existing columns
    selected_columns = [c for c in all_filtered_columns if c in df.columns]

    # filter dataframe
    df_filtered = df[selected_columns]

    return df_filtered

def clean_cu_data(df):
    """
    CU dataset has the column "light.philips_hue_lightstrip_pid_146 " 
    with white sapce at the end
    CU dataset has the column "light.philips_hue_lightstrip_pid_146 " with all NaN
    By default, df.dropna() removes any row that has at least one NaN. (we dont want that)
    """
    print("Cleaning CU data...")
    df.columns = df.columns.str.strip()
    
    # Replace fake values with NaN
    # e.g., sensor.aqara_wireless_switch_pid_081_action → mean = -9.68
    df.replace([-9.21, -9.68, -9.7, -9.81], pd.NA, inplace=True)
    
    # separate timestamp
    timestamp_col = None

    if "Timestamp" in df.columns:
        timestamp_col = df["Timestamp"]
        df = df.drop(columns=["Timestamp"])

    # keep only numeric columns
    df = df.select_dtypes(include=['number'])

    # remove near-constant sensors
    # e.g., switch.kasa_023 → std = 0, this brings ZERO information
    df = df.loc[:, df.std() > 1e-6]

    # restore timestamp
    if timestamp_col is not None:
        df.insert(0, "Timestamp", timestamp_col)

    df = df.dropna(axis=1, how="all")   # remove dead sensors
    # df = df.fillna(method="ffill")      # or interpolate
    return df

def downsample_data(df, config):
    downsample_freq = config["dataset"]["downsample_freq"]
    """
    Downsample data to target frequency while preserving events.
    continuous sensors → mean
    binary/event sensors → max
    """
    print(f"Downsampling data to {downsample_freq}...")

    # Ensure Timestamp exists
    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column not found in dataframe")

    # Convert timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Set index
    df = df.set_index("Timestamp")

    # Detect sensor types AFTER timestamp handling
    binary_cols, continuous_cols = detect_sensor_types(df)

    binary_cols = binary_cols or []
    continuous_cols = continuous_cols or []

    # aggregation dictionary
    agg_dict = {}

    for col in continuous_cols:
        agg_dict[col] = "mean"

    for col in binary_cols:
        agg_dict[col] = "max"

    # downsample
    df_down = df.resample(downsample_freq).agg(agg_dict)

    df_down = df_down.reset_index()

    return df_down

def split_actor_periods(df, config):
    """
    Split dataset into train/val/test according to actor timelines.
    - actor1_train (normal training from Actor 1 timeline 1 only)
    - actor1_val (normal validation from Actor 1 timeline 1 only)
    - actor2_test (test from Actor 2 timeline)
    - act
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Load timelines and val_ratio from config
    timelines = load_actor_timelines(config)
    val_ratio = config["dataset"]["val_ratio"]

    # Slice data according to timelines
    actor1_t1 = df[
        (df["Timestamp"] >= timelines["actor1_t1"][0]) &
        (df["Timestamp"] <= timelines["actor1_t1"][1])
    ]

    actor2 = df[
        (df["Timestamp"] >= timelines["actor2"][0]) &
        (df["Timestamp"] <= timelines["actor2"][1])
    ]

    actor1_t2 = df[
        (df["Timestamp"] >= timelines["actor1_t2"][0]) &
        (df["Timestamp"] <= timelines["actor1_t2"][1])
    ]

    # Sort
    actor1_t1 = actor1_t1.sort_values("Timestamp")
    actor2_test_df = actor2.sort_values("Timestamp")
    actor1_test_df = actor1_t2.sort_values("Timestamp")

    # Split train/val
    split_idx = int(len(actor1_t1) * (1 - val_ratio))
    train_df = actor1_t1.iloc[:split_idx]
    val_df   = actor1_t1.iloc[split_idx:]

    return train_df, val_df, actor2_test_df, actor1_test_df

def normalize(train_df, val_df, actor2_test_df, actor1_test_df, devices):
    """
    Normalize data using ONLY train_df (Actor1 timeline1)
    Returns:
    - train/val/test arrays (features only)
    - scaler
    """
    scaler = MinMaxScaler()

    # Fit scaler only on training data (features only without timestamp)
    # .to_numpy() is safer than .values() which removes column structure
    train_array = scaler.fit_transform(train_df[devices].to_numpy())
    val_array   = scaler.transform(val_df[devices].to_numpy())
    actor2_test_array = scaler.transform(actor2_test_df[devices].to_numpy())
    actor1_test_array = scaler.transform(actor1_test_df[devices].to_numpy())

    return train_array, val_array, actor2_test_array, actor1_test_array, scaler

def load_one_data(config):
    project_root_dir = config["project_root_dir"]
    dataset_name = config["dataset"]["dataset_name"]
    dataset_folder = config["dataset"]["dataset_folder"]
    data_path = f"{project_root_dir}/{dataset_folder}/{dataset_name}Master.csv"

    print(f"Loading {dataset_name} data...")
    df = pd.read_csv(data_path)
    return df

def load_and_merge_bre_cu(config):
    
    project_root_dir = config["project_root_dir"]
    dataset_folder = config["dataset"]["dataset_folder"]

    bre_data_path = f"{project_root_dir}/{dataset_folder}/BREMaster.csv"
    cu_data_path = f"{project_root_dir}/{dataset_folder}/CUMaster.csv"

    print("Loading BRE dataset...")
    bre_df = pd.read_csv(bre_data_path)

    print("Loading CU dataset...")
    cu_df = pd.read_csv(cu_data_path)

    clean_cu_data(cu_df)
    # Convert timestamp
    bre_df["Timestamp"] = pd.to_datetime(bre_df["Timestamp"])
    cu_df["Timestamp"] = pd.to_datetime(cu_df["Timestamp"])

    # Sort for time series merge
    bre_df = bre_df.sort_values("Timestamp")
    cu_df = cu_df.sort_values("Timestamp")

    print("Merging BRE+CU data...")

    merged = pd.merge_asof(
        bre_df,
        cu_df,
        on="Timestamp",
        direction="nearest"
    )

    return merged

def data_preprocessing(config):
    """
    Full preprocessing pipeline for GDN:
    1. Load CSV data (either BRE or CU or both merged)
    2. Filter data: keep selected columns only 
    3. Clean data (CU only)
    4. Downsample data to target frequency
    5. Split actor timelines
    6. Normalize features
    Returns:
    - train/val/test arrays (features only)
    - scaler
    - devices list
    - timestamp arrays (for plotting/reference)
    """
   
    merge_bre_cu = config["dataset"]["merge_bre_cu"]
    dataset_name = config["dataset"]["dataset_name"]

    # 1. Load CSV data (either BRE or CU or both merged)
    if merge_bre_cu: df = load_and_merge_bre_cu(config) # both BRE and CU
    else: df = load_one_data(config) # one data 
    
    # 2. Filter data: keep selected columns only
    if merge_bre_cu: df = filter_columns_merged_data(df, config)
    else: df = filter_columns_one_data(df, config)

    # 2. Clean CU data if loaded alone (otherwise it will cleaned in merging function)
    if not(merge_bre_cu) and dataset_name == "CU": df = clean_cu_data(df)
    
    # 4. Downsample data to target frequency
    df = downsample_data(df, config)
    

    # 5. Get device columns (exclude Timestamp)
    devices = [c for c in df.columns if c != "Timestamp"]
    print("nbr of devices:", len(devices))
    # 6. Split actors / train-val-test
    train_df, val_df, actor2_test_df, actor1_test_df = split_actor_periods(df, config)
    print("Actor split done.")

    # df.describe().to_csv("data/df_describe.csv")
    # train_df.describe().to_csv("data/train_describe.csv")
    # actor2_test_df.describe().to_csv("data/actor2_test_describe.csv")

    # 7. Save timestamps for reference/plotting
    timestamps_train = train_df['Timestamp'].to_numpy()
    timestamps_val   = val_df['Timestamp'].to_numpy()
    timestamps_actor2_test = actor2_test_df['Timestamp'].to_numpy()
    timestamps_actor1_test = actor1_test_df['Timestamp'].to_numpy()

    # 8. Normalize features
    train_array, val_array, actor2_test_array, actor1_test_array, scaler = normalize(
        train_df, val_df, actor2_test_df, actor1_test_df, devices
    )
    print("Normalization done.")

    return (train_array, val_array, actor2_test_array, actor1_test_array,
            scaler, devices,
            timestamps_train, timestamps_val, timestamps_actor2_test, timestamps_actor1_test)