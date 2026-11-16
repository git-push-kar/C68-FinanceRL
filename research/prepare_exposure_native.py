"""One-time preparation of an isolated native exposure branch; does not run it."""
from pathlib import Path
ROOT=Path(__file__).parent;branch=ROOT/'rl_exposure_native_trial'
def replace(text,old,new):
    if text.count(old)!=1:raise AssertionError(f'Expected unique source fragment: {old[:100]}')
    return text.replace(old,new)
p=branch/'text_policy.py';s=p.read_text()
s=replace(s,'from selected_readout import selected_readout','from selected_readout import selected_readout\nfrom exposure_prompt import prompt as exposure_prompt')
s=s.replace("'sector_text_v1'","'exposure_text_v1'")
start=s.index('    def prompt(self,obs):');end=s.index('    def forward(self,obs):',start)
s=s[:start]+"    def prompt(self,obs):\n        return exposure_prompt(obs,self.tickers)\n"+s[end:]
s=replace(s,'    def forward(self,obs):','    def action_logits(self,obs):')
s=replace(s,"if obs['market'].shape[0]!=1:","if obs['summary'].shape[0]!=1:")
s=replace(s,'        # Fixed exploration temperature, identical in the base and tuned policies.','        return logits\n    def forward(self,obs):\n        logits=self.action_logits(obs)\n        # Fixed exploration temperature, identical in the base and tuned policies.')
s=replace(s,'return torch.distributions.Categorical(probs=probabilities),h.new_zeros(1)','return torch.distributions.Categorical(probs=probabilities),logits.new_zeros(1)')
p.write_text(s)
p=branch/'train_counterfactual.py';s=p.read_text()
s=replace(s,'from strategy_ppo import Market,evaluate,tensor,NAMES','from exposure_market import Market,evaluate,tensor,NAMES\nfrom soft_improvement import actor_loss')
s=replace(s,"'algorithm':'all_action_truncated_return_policy_gradient'","'algorithm':'native_soft_fitting_truncated_Monte_Carlo_policy_iteration'")
s=replace(s,"'reward':'100 * net log wealth excess over daily equal-weight, same as earlier PPO reward'","'reward':'100*(daily net return - 1.5*daily net return squared); fixed risk aversion 3'")
s=replace(s,"entropy coefficient=.005, gradient norm cap=1","soft target temperature 1, cross entropy to softmax of normalized MC values, gradient norm cap=1")
s=replace(s,"'limitations':'Finite-horizon Monte Carlo values have truncation bias. No claim of exact full-horizon gradient or novel all-action estimator. Reward, horizon and budget differ from the GRPO feasibility pilot.'","'limitations':'Approximate finite-horizon Monte Carlo policy iteration; noisy nonlinear targets and no future entropy in values. Not exact SAC/MPO, unbiased full-horizon gradient or a novelty claim.'")
s=replace(s,"'prior_art':'https://jmlr.org/papers/v21/18-012.html'","'prior_art':['https://arxiv.org/abs/1806.06920','https://arxiv.org/abs/1801.01290']")
s=replace(s,"'cost':'10bp * full traded-stock L1; five-session decisions; drift; one-close market-signal delay'","'cost':'Exact self-financing fees: 10bp per unit traded stock; five-session decisions; zero-interest cash; one-close signal lag'")
s=replace(s,"'evaluation':'Same expected-strategy-mixture rule before and after RL'","'evaluation':'Same expected equity exposure before and after RL'")
s=replace(s,"wait_for_compute_window();dist,_=policy(tensor(obs,'cuda'))","wait_for_compute_window();logits=policy.action_logits(tensor(obs,'cuda'))\n            dist=torch.distributions.Categorical(probs=.95*torch.softmax(logits/2.,-1)+.05/4)")
s=replace(s,"loss=(-(dist.probs[0]*torch.as_tensor(adv,device='cuda')).sum()-.005*dist.entropy().mean())/batch_states","loss=actor_loss(logits,torch.as_tensor(adv,device='cuda'))/batch_states")
p.write_text(s)
for name in ('audit_text_run.py','text_gate.py'):
    p=branch/name;s=p.read_text();s=s.replace('from strategy_ppo import Market','from exposure_market import Market')
    if name=='text_gate.py':
        s=replace(s,"Path('../checks/readout_lora_rl_gate')","Path('../checks/exposure_native_soft_gate')")
        start=s.index("    reference=Path('../checks/native_text_rl_gate/result.json')")
        end=s.index('    trainable_count=',start)
        s=s[:start]+"    with torch.no_grad(),policy.shared.model.disable_adapter():\n        original=policy(input)[0].probs[0].cpu().numpy()\n    base_reference_error=float(np.max(abs(np.array(initial_probabilities)-original)))\n    if base_reference_error>1e-7:raise AssertionError('Initial adapter changed original exposure policy')\n"+s[end:]
    p.write_text(s)
print('Prepared isolated native exposure source; no GPU job launched.')
