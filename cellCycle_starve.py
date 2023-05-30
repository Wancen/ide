from scipy.stats import zscore
import pandas as pd
from sync_models import *
# df = pd.read_csv('../data/cellCycle/cell_cycle_Starve.csv', header=None, names=['col{}'.format(i) for i in range(1, 48)])
df = pd.read_csv('../data/cellCycle/20211210-1445_knn150_t10_gamma0.25_allconditions.csv')
df_Starve = df[["AreaShape_Area_cell","Int_Med_cycD1_nuc",
"AreaShape_Area_nuc",	"Int_Med_cycE_nuc",
"Int_Intg_DNA_nuc",	"Int_Med_p16_nuc",
"Int_Med_BP1_nuc",	"Int_Med_p21_nuc",
"Int_Med_CDK11b_nuc",	"Int_Med_p27_nuc",
"Int_Med_CDK2_nuc",	"Int_Med_p38_nuc",
"Int_Med_CDK4_nuc",	"Int_Med_p53_nuc",
"Int_Med_CDK6_nuc",	"Int_Med_pCHK1_nuc",
"Int_Med_Cdh1_nuc",	"Int_Med_pCdc6_nuc",
"Int_Med_Cdt1_nuc",	"Int_Med_pH2AX_nuc",
"Int_Med_DNA_nuc",	"Int_Med_pRB_nuc",
"Int_Med_E2F1_nuc",	"Int_Med_pS6_nuc",
"Int_Med_Fra1_nuc",	"Int_Med_pp21_nuc",
"Int_Med_RB_nuc",	"Int_Med_pp27_nuc",
"Int_Med_S6_nuc",	"Int_Med_pp38_nuc",
"Int_Med_Skp2_nuc",	"Int_Med_pp53_nuc",
"Int_Med_YAP_nuc",	"Int_Med_PCNA_nuc",
"Int_Med_cFos_nuc",	"Int_Std_PCNA_nuc",
"Int_Med_cJun_nuc",	"Int_pRB_over_RB_nuc",
"Int_Med_cMyc_nuc",	"Int_pS6_over_S6_nuc",
"Int_Med_cdc6_nuc",	"Int_pp21_over_p21_nuc",
"Int_Med_cycA_nuc",	"Int_pp27_over_p27_nuc",
"Int_Med_cycB1_nuc",	"Int_pp53_over_p53_nuc",
	"Int_pp38_over_p38_nuc"]]
df['Condition'].unique()
df_Starve = df_Starve[df["Condition"].str.contains("Starve")]

df_standardized = df_Starve.apply(zscore)

device = select_device()
nepoch = 50000
seed = 123

d = df_Starve.shape[1]
t = 2
kappa = 20
bs = 1000
lr = 0.005
coefs = [-20, -10, 0]
# c = torch.ones(df_Starve.shape[0], device= device).unsqueeze(1)
numeric_values = [int(s.split('-')[1]) for s in df["Condition"][df["Condition"].str.contains("Starve")].values]
for i in range(len(numeric_values)):
    if numeric_values[i] == 1:
        # If the element is 1, set it to 0
        numeric_values[i] = 0
    elif numeric_values[i] == 7:
        # If the element is 7, set it to 1
        numeric_values[i] = 1


c = torch.tensor(torch.nn.functional.one_hot(torch.tensor(numeric_values), t), device=device)
c = c.to(torch.float32)
x = torch.tensor(df_standardized.values, device= device)
x = x.to(torch.float32)

for coef in coefs:
    save_path = f"cellCycle/param_prior_relu/Starve/{coef}_{kappa}_{t}"
    os.makedirs(save_path, exist_ok=True)
    encoder = Syn_Encoder(d, t, kappa).to(device)
    generator = Syn_Generator(d, t, kappa).to(device)
    prior = ParamPrior(t, kappa).to(device)
    loggamma = nn.Parameter(coef*torch.ones(1, device=device))
    opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma]+list(prior.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch, cycle_momentum=False)
    t1 = time.time()
    for epoch in range(nepoch):
        muc, logvarc = prior(c)
        mean, logvar = encoder(x, c)
        postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
        kl = torch.sum(torch.exp(logvar - logvarc) + torch.square(mean - muc) / torch.exp(logvarc) - logvar + logvarc, dim=1) - kappa
        xhat = generator(postz, c)
        recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
        recon2 = d * loggamma + math.log(2 * math.pi) * d
        loss = torch.mean(recon1 + recon2 + kl)
        gamma = torch.exp(loggamma)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if (epoch+1) % 2000 == 0:
            print(f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1*gamma)}, gamma= {gamma.data}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")
    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")
    torch.save(loggamma, f"{save_path}/loggamma.pth")
    torch.save(prior, f"{save_path}/prior.pth")

ad_dot1 = []
ad_dot05 = []
for coef in coefs:
    save_path = f"cellCycle/param_prior_relu/Starve/{coef}_{kappa}_{t}"
    encoder = torch.load(f"{save_path}/encoder.pth")
    prior = torch.load(f"{save_path}/prior.pth")
    xt = x
    ct = c
    muc, logvarc = prior(ct)
    mean, logvar = encoder(xt, ct)
    var_ratio = torch.exp(logvar - logvarc).mean(dim=0)
    print(var_ratio)
    count1 = (var_ratio < 0.1).sum()
    ad_dot1.append(count1.tolist())
    count2 = (var_ratio < 0.05).sum()
    ad_dot05.append(count2.tolist())
np.save(f"{save_path}/../Starve_cvae_dot1.npy", np.array(ad_dot1))
np.save(f"{save_path}/../Starve_cvae_dot05.npy", np.array(ad_dot1))

# save_path = f"cellCycle/param_prior_relu/"
# np.load(f"{save_path}/ad_dot05.npy")
# np.load(f"{save_path}/ad_dot1.npy")