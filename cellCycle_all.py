from scipy.stats import zscore
import pandas as pd
from sync_models import *
# df = pd.read_csv('../data/cellCycle/cell_cycle_all.csv', header=None, names=['col{}'.format(i) for i in range(1, 48)])
df = pd.read_csv('../data/cellCycle/20211210-1445_knn150_t10_gamma0.25_allconditions.csv')
df_all = df[["AreaShape_Area_cell","Int_Med_cycD1_nuc",
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
class_value = df['Condition'].unique()

df_standardized = df_all.apply(zscore)

device = select_device()
nepoch = 50000
seed = 123

d = df_all.shape[1]
t = len(class_value)
kappa = 20
bs = 1000
lr = 0.005
coefs = [-20, 0]
# c = torch.ones(df_all.shape[0], device= device).unsqueeze(1)
class_map = {'Ctl': 0, 'Etop-1': 1, 'Etop-2': 2, 'Etop-3': 3, 'Etop-4': 4, 'H2O2-1': 5, 'H2O2-2': 6,
    'H2O2-3': 7, 'Starve-1' : 8, 'Starve-7': 9}
df['Condition2'] = df['Condition'].map(class_map)
c = torch.tensor(torch.nn.functional.one_hot(torch.tensor(df['Condition2'].values), t), device=device)
c = c.to(torch.float32)
x = torch.tensor(df_standardized.values, device= device)
x = x.to(torch.float32)

for coef in coefs:
    save_path = f"cellCycle/param_prior_relu/all/{coef}_{kappa}_{t}"
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
    save_path = f"cellCycle/param_prior_relu/all/{coef}_{kappa}_{t}"
    encoder = torch.load(f"{save_path}/encoder.pth")
    prior = torch.load(f"{save_path}/prior.pth")
    loggamma = torch.load(f"{save_path}/loggamma.pth")
    xt = x
    ct = c
    muc, logvarc = prior(ct)
    mean, logvar = encoder(xt, ct)
    np.save(f"{save_path}/meanz.npy", np.array(mean.cpu().detach()))
    var_ratio = torch.exp(logvar - logvarc).mean(dim=0)
    np.save(f"{save_path}/var_ratio.npy", np.array(var_ratio.cpu().detach()))
    print(var_ratio)
    count1 = (var_ratio < 0.1).sum()
    ad_dot1.append(count1.tolist())
    count2 = (var_ratio < 0.05).sum()
    ad_dot05.append(count2.tolist())
np.save(f"{save_path}/../all_cvae_dot1.npy", np.array(ad_dot1))
np.save(f"{save_path}/../all_cvae_dot05.npy", np.array(ad_dot1))

# save_path = f"cellCycle/param_prior_relu/"
# np.load(f"{save_path}/ad_dot05.npy")
mean = np.load(f"{save_path}/meanz.npy")
import matplotlib.pyplot as plt
plt.scatter(mean[:,9], df['Condition2'].values)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatterplot')

# Save the scatterplot as a PNG file
plt.savefig('scatterplot.png')