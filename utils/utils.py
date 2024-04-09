import numpy as np
import torch
def testwrite(loader,outfile=outfile):
    count=0
    model.eval()
    seperator=";"
    y_available=True
    loss_all=0
    with open(outfile, "w+") as g:
        g.write("")
    for data in loader:
      count+=1
      y_output=[]
      ylabel=[]
      entropy=[]
      latentnames=[]
      img_pos=[]
      error = 0
      data = [im for im in data.values()]
      images=data[0].to(device).float()
      aug=data[5].to(device).float()
      labels=data[3].to(device)
      images = torch.cat([images, aug], dim=0)
      with torch.no_grad():
          features = model(images)
          f1, f2 = torch.split(features, [bs, bs], dim=0)
          features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
          loss = criterion(features)  
      if y_available:
              y_output.append(features.cpu().detach().numpy())
              ylabel.append(labels.cpu().detach().numpy())
              entropy.append(loss.cpu().detach().numpy())
              img_pos.append(data[2])
      with open(outfile, "a+") as g:
          for no,point in enumerate(data[4]):
              youtput_str=[str(value) for value in y_output[0][no][0]]
              g.writelines(point+seperator+str(ylabel[0][no])+seperator+str(entropy[0]))
              g.write(";")
              g.write(";".join(youtput_str))
              g.write("\n")
    return loss_all/(len(loader)*bs)

