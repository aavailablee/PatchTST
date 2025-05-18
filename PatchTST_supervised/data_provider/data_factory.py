from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from data_provider.scaler import init_scaler
from data_provider.data_loader import AttrMapper, BSIDMapper, Ice
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

scaler = init_scaler("standard")  # 假设这里是正确的scaler初始化方式
scalery = init_scaler("standard") 
def data_provider(args, flag):
    if args.data == "ice":
        # ==== 1. 参数校验 ====
        valid_flags = ['train', 'val', 'test']
        if flag not in valid_flags:
            raise ValueError(f"Invalid flag: {flag}. Must be one of {valid_flags}.")
        
        # ==== 2. 初始化配置和转换器 ====
        # (保持原有配置逻辑，但按需加载)
        transformer = AttrMapper()
        id_transformer = BSIDMapper()
        
        # ==== 3. 动态加载单个数据集 ====
        cfg_dict = {
            "dataset": {
                "have_weather_forecast": False,
                "data_path": "./dataset/all_ice/",
            },
            "batch_size": 64,
            "num_workers": 1,
        }
        
        # 将配置字典转换为对象
        class ConfigObj:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, ConfigObj(value))
                    else:
                        setattr(self, key, value)
        cfg = ConfigObj(cfg_dict)
        
        # ==== 4. 仅创建flag对应的数据 ====
        dataset = Ice(
            cfg=cfg.dataset,
            label=flag,  # 根据传入的flag动态选择
            transformer=transformer,
            id_transformer=id_transformer,
            scaler=scaler,
            scalerY=scalery
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(flag == 'train'),  # 仅在训练时shuffle
            num_workers=cfg.num_workers
        )
        
        # ==== 5. 返回与flag对应的数据 ====
        return dataset, dataloader
        return dataset, dataloader, scaler, dataset.bsid  # 直接返回对应的bsid
        # scaler = init_scaler("standard")
        # dataloaders = {}

        # transformer = AttrMapper()
        # id_transformer = BSIDMapper()
        # # weat_info_true: False  # 是否加载天气信息
        # # weat_dim: 3  # 输入维度（天气）
        # # attribute_true: True  # 是否添加导体属性信息
        # # topo_true: False  # 是否添加地形类别
        # cfg = {
        #     "dataset": {
        #         "have_weather_forecast": False,
        #         "data_path": "../dataset/all_ice/",
                
        #     },
        #     "batch_size": 64,
        #     "num_workers": 1,
        # }

        # class obj(object):
        #     def __init__(self, d):
        #         for k, v in d.items():
        #             if isinstance(k, (list, tuple)):
        #                 setattr(
        #                     self,
        #                     k,
        #                     [obj(x) if isinstance(x, dict) else x for x in v],
        #                 )
        #             else:
        #                 setattr(self, k, obj(v) if isinstance(v, dict) else v)

        # cfg = obj(cfg)
        # for category in ["train", "valid", "test"]:
        #     dataset = Ice(cfg.dataset, category, transformer, id_transformer,scaler)
        #     dataloaders[category] = DataLoader(
        #         dataset,
        #         batch_size=cfg.batch_size,
        #         shuffle=True if category == "train" else False,
        #         num_workers=cfg.num_workers,
        #     )
        #     if category == "test":
        #         d=dataset.bsid
        # train, valid, test = (
        #     dataloaders["train"],
        #     dataloaders["valid"],
        #     dataloaders["test"],
        # )
        # return train, valid, test, scaler, d
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    '''
    drop_last，如果最后一批数量小于batchSize，丢弃，适合训练逻辑
    shuffle_flag，打乱顺序，避免学习顺序特征，提高泛化能力
    '''
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
