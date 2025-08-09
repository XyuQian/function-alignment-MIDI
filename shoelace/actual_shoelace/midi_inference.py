import os
import torch
import torch.nn.functional as F
import numpy as np
from shoelace.midi_lm.models.config import PAD, SEG_RES, RES_EVENT
from shoelace.datasets.preprocess_midi import load_midi
from shoelace.datasets.utils import decode
from shoelace.utils.network_utils import transform_inputs
from shoelace.actual_shoelace.midi_config import TASKS, MODEL_MAPPING


def remove_head(x):
    idx = x[0, :, 0] < RES_EVENT
    return x[0, idx].unsqueeze(0)


def cut_midi(input_ids, hop_len, chunk_len):
    input_ids = input_ids[input_ids[:, 0] < RES_EVENT]

    seg_pos = torch.arange(len(input_ids)).to(input_ids.device)
    seg_pos = seg_pos[input_ids[:, 0] == SEG_RES]
    prefix = input_ids[:seg_pos[hop_len]]
    suffix = input_ids[seg_pos[hop_len]: seg_pos[chunk_len] + 1]
    
    sustain = hop_len + 1
    res_events = []
    for i, event in enumerate(prefix):
        if event[0] == SEG_RES:
            sustain -= 1
            continue

        if event[3] >= sustain:
            new_event = event + 0
            new_event[0] = RES_EVENT
            new_event[3] = event[3] - sustain
            res_events.append(new_event)

    if len(res_events) == 0:
        return prefix, suffix
    res_event = torch.stack(res_events, 0)
    return prefix, torch.concat([suffix[:1], res_event, suffix[1:]], 0)


def get_midi_data(path, chunk_frame, hop_frame, device):
    results = load_midi(path)
                    
    assert results is not None
    events = results["events"]
    sos = results["sos"]
    res_events = results["res_events"]
    res_sos = results["res_sos"]

    chunk_len = int(chunk_frame//SEG_RES)
    hop_len = int(hop_frame//SEG_RES)
    for st_id in range(0, len(sos) - chunk_len, hop_len):
        event_st_id = sos[st_id]
        event_ed_id = sos[st_id + chunk_len]
        
        if st_id + 1 < len(res_sos):
            res_st_id = res_sos[st_id]
            res_ed_id = res_sos[st_id + 1]
        else:
            res_st_id = res_ed_id = 0
        
        seq = events[event_st_id : event_ed_id + 1]
        if res_st_id < res_ed_id:
            seq = np.concatenate([seq[:1], res_events[res_st_id : res_ed_id], seq[1:]], 0)
        input_ids = torch.from_numpy(seq).long().unsqueeze(0).to(device)
        input_ids[input_ids < 0] = PAD
        midi_index = transform_inputs(input_ids[..., 0], SEG_RES).long().to(device)
        midi_index = F.pad(midi_index, (1, 0), "constant", 0)
        
        yield input_ids, midi_index

    seq = torch.from_numpy(events[:event_ed_id]).unsqueeze(0)
    # seq = torch.from_numpy(events).unsqueeze(0)
    seq[seq < 0] = PAD
    seq = seq.to(device)
    yield seq, None


def get_full_midi_data(path, device):
    results = load_midi(path)
    MAX_SEQ_LEN = 1024
                    
    assert results is not None
    events = results["events"]
    sos = results["sos"]
    res_events = results["res_events"]
    res_sos = results["res_sos"]

    seq = events[:sos[-1]]
    seq[seq < 0] = PAD
    seq = seq[:MAX_SEQ_LEN]
    input_ids = torch.from_numpy(seq).long().unsqueeze(0).to(device)
    midi_index = transform_inputs(input_ids[..., 0], SEG_RES).long().to(device)
    midi_index = F.pad(midi_index, (1, 0), "constant", 0)
    
    return input_ids, midi_index


class InferenceHelper:
    def __init__(self, model_folder, n_prompts, task_type, mask_config, device):
        from shoelace.actual_shoelace.midi_shoelace import Shoelace as Model
        from shoelace.actual_shoelace.midi_config import MODEL_FACTORY
        self.model = Model(
            device=device, 
            n_prompts=n_prompts,
            model_configs=MODEL_FACTORY, 
            task_type=task_type,
            mask_config=mask_config
        )
        self.model.load_weights(model_folder)
        self.model.eval().to(device)
        self.mask_config = mask_config
        self.device = device

    @torch.no_grad()
    def score_2_perf(self, midi_path, max_gen_len, top_k, tasks):
        input_ids, score_index = get_full_midi_data(midi_path, device=self.device)
        print(f"Input shape: {input_ids.shape}")
        print(f"Score index shape: {score_index.shape}")

        # Use ScoreLM to populate cache
        score_gen = self.model.inference(
            model_name="ScoreLM",
            cond_model_name=None,
            max_len=(input_ids[0, :, 0] == SEG_RES).sum(),
            use_generator=False,
            reset_cond_cache=True,
            last_chunk=True,
            input_ids=input_ids,
            top_k=top_k,
            tasks=[tasks[0]],
            device=self.device
        )
        print("Score generation shape:", score_gen.shape)

        perf_codes = self.model.inference(
            model_name="PerformanceLM",
            cond_model_name="ScoreLM",
            max_len=max_gen_len,
            use_generator=True,
            reset_cond_cache=False,
            last_chunk=True,
            input_ids=None,
            cond_indices=score_index,
            top_k=top_k,
            tasks=[tasks[1]],
            batch_size=len(input_ids),
            device=self.device
        )
        print("Performance generation shape:", perf_codes.shape)

        return perf_codes, score_gen
    
    @torch.no_grad()
    def perf_2_score(self, midi_path, max_gen_len, top_k, tasks):
        input_ids, perf_index = get_full_midi_data(midi_path, device=self.device)
        print(f"Input shape: {input_ids.shape}")
        print(f"Performance index shape: {perf_index.shape}")

        # Use PerformanceLM to populate cache
        perf_gen = self.model.inference(
            model_name="PerformanceLM",
            cond_model_name=None,
            max_len=(input_ids[0, :, 0] == SEG_RES).sum(),
            use_generator=False,
            reset_cond_cache=True,
            last_chunk=True,
            input_ids=input_ids,
            top_k=top_k,
            tasks=[tasks[0]],
            device=self.device
        )
        print("Performance generation shape:", perf_gen.shape)

        score_codes = self.model.inference(
            model_name="ScoreLM",
            cond_model_name="PerformanceLM",
            max_len=max_gen_len,
            use_generator=True,
            reset_cond_cache=False,
            last_chunk=True,
            input_ids=None,
            cond_indices=perf_index,
            top_k=top_k,
            tasks=[tasks[1]],
            batch_size=len(input_ids),
            device=self.device
        )
        print("Score generation shape:", score_codes.shape)

        return score_codes, perf_gen
    
    @torch.no_grad()
    def direct_gen(self, midi_path, max_gen_len, top_k):
        input_ids, midi_index = get_full_midi_data(midi_path, device=self.device)
        print(f"Input shape: {input_ids.shape}")
        print(f"MIDI index shape: {midi_index.shape}")

        score_lm = self.model.model_dict["ScoreLM"]["model_obj"]
        perf_lm = self.model.model_dict["PerformanceLM"]["model_obj"]

        score_lm.reset_cache()
        score_lm.set_use_generator(False)
        score_gen = score_lm.inference(
            input_ids,
            max_len=max_gen_len,
            top_k=top_k,
            last_chunk=True
        )
        perf_lm.reset_cache()
        perf_lm.set_use_generator(False)
        perf_gen = perf_lm.inference(
            input_ids,
            max_len=max_gen_len,
            top_k=top_k,
            last_chunk=True
        )
        
        return score_gen, perf_gen


if __name__=="__main__":
    # inference_helper_score2perf = InferenceHelper(
    #     model_folder="exp/midi_conversion/latest_100_end_score_2_perf", 
    #     device=torch.device("cuda"),
    #     n_prompts=5,
    #     task_type="midi_conversion",
    #     mask_config={
    #         "ScoreLM": False,
    #         "PerformanceLM": True
    #     }
    # )
    
    inference_helper_perf2score = InferenceHelper(
        model_folder="exp/perf_2_score_bs32_ep100_mask0.2_estop/latest_20_6153_perf_2_score",
        # model_folder="exp/perf_2_score_bs64_ep50_mask0.5/latest_50_end_perf_2_score",
        device=torch.device("cuda"),
        n_prompts=5,
        task_type="midi_conversion",
        mask_config={
            "ScoreLM": True,
            "PerformanceLM": False
        }
    )


    # Example usage
    fname = "001_001"
    # fname = "2"
    # score_data_generator = get_midi_data(
    #     path=f"data/ASAP/ASAP_samples/Score/{fname}.mid", 
    #     chunk_frame=512, 
    #     hop_frame=256, 
    #     device=torch.device("cuda")
    # )
    # perf_data_generator = get_midi_data(
    #     path=f"data/ASAP/ASAP_samples/Performance/{fname}.mid", 
    #     chunk_frame=512, 
    #     hop_frame=256, 
    #     device=torch.device("cuda")
    # )

    # print(f"\n********** Score to Performance for {fname} ***********")
    # perf_codes, score_gen = inference_helper_score2perf.score_2_perf(
    #     midi_path=f"data/ASAP/Score/{fname}.mid",
    #     # midi_path=f"data/{fname}.midi",
    #     max_gen_len=128,
    #     top_k=1, 
    #     tasks=['generate_score', 'generate_performance']
    # )
    # print(perf_codes.shape, score_gen.shape)
    # print()

    print(f"********** Performance to Score for {fname} ***********")
    score_codes, perf_gen = inference_helper_perf2score.perf_2_score(
        midi_path=f"data/ASAP/Performance/{fname}.mid",
        # midi_path=f"data/{fname}.midi",
        max_gen_len=128,
        top_k=4, 
        tasks=['generate_performance', 'generate_score']
    )
    print(score_codes.shape, perf_gen.shape)
    print()

    # decode(os.path.join("inference_results", f"sanity_{fname}_score.mid"), perf_codes[0].cpu().numpy())
    # decode(os.path.join("inference_results", f"sanity_{fname}_perf.mid"), score_codes[0].cpu().numpy())

    # decode(os.path.join("inference_results", f"sanity_{fname}_score_gen.mid"), score_gen[0].cpu().numpy())
    # decode(os.path.join("inference_results", f"sanity_{fname}_perf_gen.mid"), perf_gen[0].cpu().numpy())
    # decode(os.path.join("inference_results", f"{fname}_score_2_perf.mid"), perf_codes[0].cpu().numpy())
    decode(os.path.join("inference_results", f"{fname}_perf_2_score.mid"), score_codes[0].cpu().numpy())

    # decode(os.path.join("inference_results", f"with_prompt_{fname}_score_2_perf.mid"), perf_codes[0].cpu().numpy())
    # decode(os.path.join("inference_results", f"with_prompt_{fname}_perf_2_score.mid"), score_codes[0].cpu().numpy())
    # decode(os.path.join("inference_results", f"sanity_{fname}_score_input.mid"), input_score[0].cpu().numpy())
    # decode(os.path.join("inference_results", f"sanity_{fname}_perf_input.mid"), input_perf[0].cpu().numpy())

    # print(f"********** Direct generation for {fname} ***********")
    # score_codes, _ = inference_helper_score2perf.direct_gen(
    #     midi_path=f"data/ASAP/Score/{fname}.mid",
    #     max_gen_len=128,
    #     top_k=1
    # )

    # _, perf_codes = inference_helper_perf2score.direct_gen(
    #     midi_path=f"data/ASAP/Performance/{fname}.mid",
    #     max_gen_len=128,
    #     top_k=1
    # )
    # print(score_codes.shape)
    # print(perf_codes.shape)

    # decode(os.path.join("inference_results", f"direct_gen_{fname}_score.mid"), score_codes[0].cpu().numpy())
    # decode(os.path.join("inference_results", f"direct_gen_{fname}_perf.mid"), perf_codes[0].cpu().numpy())

