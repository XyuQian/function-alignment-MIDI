import torch
from shoelace.midi_lm.models.config import SEG_RES, RES_EVENT


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

class InferenceHelper:
    def __init__(self, model_folder, n_prompts, task_type, model_type, device):
        from shoelace.actual_shoelace.shoelace import Shoelace as Model
        from shoelace.actual_shoelace.config import MODEL_FACTORY
        self.model = Model(mask_type=model_type, 
            device=torch.device("cuda"), 
            model_configs=MODEL_FACTORY, 
            task_type=task_type,
            n_prompts=n_prompts)
        self.model.load_weights(model_folder)
        self.model.eval().to(device)
        self.model_type = model_type

    
    @torch.no_grad()
    def midi_2_audio(self, input_ids_generator, chunk_frame, hop_frame, top_k, tasks):
        audio_prompt = None
        n_id = 0
        results = []
        for input_ids, midi_index in input_ids_generator:

            if midi_index is None:
                break
            # print(input_ids)
            # print(mii_index)
            self.model.inference(model_name="MIDILM", 
                            max_len=(input_ids[0, :, 0] == SEG_RES).sum(),
                            cond_model_name="AudioLM", 
                            use_generator=True, top_k=16, 
                            last_chunk=True, input_ids=input_ids, 
                            tasks=[tasks[0]],
                            reset_cache=True)


            audio_codes = self.model.inference(model_name="AudioLM", 
                            cond_model_name="MIDILM", 
                            max_len=chunk_frame - hop_frame + 4 if n_id > 0 else chunk_frame + 4,
                            use_generator=True, top_k=top_k, reset_cache=False,
                            last_chunk=False, input_ids=audio_prompt, cond_indices=midi_index,
                            batch_size=len(input_ids), 
                            tasks=[tasks[1]], device=input_ids.device)
            results.append(audio_codes[:, :hop_frame])
            audio_prompt = audio_codes[:, hop_frame:chunk_frame]
            
            
            n_id += 1
            if n_id > 3:
                break
            
        results.append(audio_prompt)
        audio_codes = torch.concat(results, 1).transpose(1, 2)
        return audio_codes, input_ids


    @torch.no_grad()
    def audio_2_midi(self, input_ids_generator, chunk_frame, hop_frame, top_k, tasks):
        midi_prompt = None
        n_id = 0
        chunk_len = chunk_frame // SEG_RES
        hop_len = hop_frame // SEG_RES
        results = []
        for input_ids, audio_index in input_ids_generator:
            if audio_index is None:
                break
            self.model.inference(model_name="AudioLM", cond_model_name="MIDILM",
                            max_len=1, input_ids=input_ids, tasks=[tasks[0]],
                            use_generator=True, top_k=top_k, reset_cache=True,
                            last_chunk=True, device=input_ids.device)


            midi_codes = self.model.inference(model_name="MIDILM", tasks=[tasks[1]],
                            cond_model_name="AudioLM", max_len=chunk_len - 2,
                            use_generator=True, top_k=top_k, reset_cache=False,
                            last_chunk=False, input_ids=midi_prompt, 
                            cond_indices=audio_index,
                            batch_size=len(input_ids), device=input_ids.device)

            prefix, midi_prompt = cut_midi(midi_codes.squeeze(0), hop_len, chunk_len - 2)
            # print("prefix", prefix)
            # print("midi_prompt", midi_prompt)
            results.append(prefix.unsqueeze(0))
            midi_prompt = midi_prompt.unsqueeze(0)
            
            n_id += 1
            if n_id > 3:
                break
        
        
        results.append(remove_head(midi_prompt))
        midi_codes = torch.concat(results, 1)
        return midi_codes, input_ids.transpose(1, 2)



if __name__=="__main__":
    inference_helper = InferenceHelper()
    input_ids = torch.ones([1, 10, 6]).cuda().long()
    audio_codes = inference_helper.inference(input_ids, max_len=16*50)
    print(audio_codes.shape)
        
            