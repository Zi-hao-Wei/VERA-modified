# Visualization
if itr % args.viz_every == 0:
    del x_g, h_g
    x_g, h_g = g.sample(args.batch_size, requires_grad=True)

    plot(("{}/{:0%d}_init.png" % niters_digs).format(data_sgld_dir, itr),
            x_g.view(x_g.size(0), *args.data_size))

    if args.mog_comps is not None or args.nice:
        x_mog = logp_net.sample(args.batch_size)
        plot(("{}/{:0%d}_MOG.png" % niters_digs).format(data_sgld_dir, itr),
                x_mog.view(x_mog.size(0), *args.data_size))

    # input space sgld
    x_sgld = x_g
    steps = [x_sgld.detach()]
    accepts = []
    for k in range(args.sgld_steps):
        [x_sgld], a = utils.hmc.MALA([x_sgld], lambda x: logp_net(x).squeeze(), sgld_lr)
        steps.append(x_sgld.detach())
        accepts.append(a.item())
    ar = np.mean(accepts)
    utils.print_log("data accept rate: {}".format(ar), args)
    sgld_lr = sgld_lr + args.mcmc_lr * (ar - .57) * sgld_lr
    plot(("{}/{:0%d}_ref.png" % niters_digs).format(data_sgld_dir, itr),
            x_sgld.view(x_g.size(0), *args.data_size))

    chain = torch.cat([step[0][None] for step in steps], 0)
    plot(("{}/{:0%d}.png" % niters_digs).format(data_sgld_chain_dir, itr),
            chain.view(chain.size(0), *args.data_size))

    # latent space sgld
    eps_sgld = torch.randn_like(x_g)
    z_sgld = torch.randn((eps_sgld.size(0), args.noise_dim)).to(eps_sgld.device)
    vs = (z_sgld.requires_grad_(), eps_sgld.requires_grad_())
    steps = [vs]
    accepts = []
    gfn = lambda z, e: g.g(z) + g.logsigma.exp() * e
    efn = lambda z, e: logp_net(gfn(z, e)).squeeze()
    with torch.no_grad():
        x_sgld = gfn(z_sgld, eps_sgld)
    plot(("{}/{:0%d}_init.png" % niters_digs).format(gen_sgld_dir, itr),
            x_sgld.view(x_g.size(0), *args.data_size))
    for k in range(args.sgld_steps):
        vs, a = utils.hmc.MALA(vs, efn, sgld_lr_z)
        steps.append(vs)
        accepts.append(a.item())
    ar = np.mean(accepts)
    utils.print_log("latent eps accept rate: {}".format(ar), args)
    sgld_lr_z = sgld_lr_z + args.mcmc_lr * (ar - .57) * sgld_lr_z
    z_sgld, eps_sgld = steps[-1]
    with torch.no_grad():
        x_sgld = gfn(z_sgld, eps_sgld)
    plot(("{}/{:0%d}_ref.png" % niters_digs).format(gen_sgld_dir, itr),
            x_sgld.view(x_g.size(0), *args.data_size))

    z_steps, eps_steps = zip(*steps)
    z_chain = torch.cat([step[0][None] for step in z_steps], 0)
    eps_chain = torch.cat([step[0][None] for step in eps_steps], 0)
    with torch.no_grad():
        chain = gfn(z_chain, eps_chain)
    plot(("{}/{:0%d}.png" % niters_digs).format(gen_sgld_chain_dir, itr),
            chain.view(chain.size(0), *args.data_size))

    # latent space sgld no eps
    z_sgld = torch.randn((eps_sgld.size(0), args.noise_dim)).to(eps_sgld.device)
    vs = (z_sgld.requires_grad_(),)
    steps = [vs]
    accepts = []
    gfn = lambda z: g.g(z)
    efn = lambda z: logp_net(gfn(z)).squeeze()
    with torch.no_grad():
        x_sgld = gfn(z_sgld)
    plot(("{}/{:0%d}_init.png" % niters_digs).format(z_sgld_dir, itr),
            x_sgld.view(x_g.size(0), *args.data_size))
    for k in range(args.sgld_steps):
        vs, a = utils.hmc.MALA(vs, efn, sgld_lr_zne)
        steps.append(vs)
        accepts.append(a.item())
    ar = np.mean(accepts)
    utils.print_log("latent accept rate: {}".format(ar), args)
    sgld_lr_zne = sgld_lr_zne + args.mcmc_lr * (ar - .57) * sgld_lr_zne
    z_sgld, = steps[-1]
    with torch.no_grad():
        x_sgld = gfn(z_sgld)
    plot(("{}/{:0%d}_ref.png" % niters_digs).format(z_sgld_dir, itr),
            x_sgld.view(x_g.size(0), *args.data_size))

    z_steps = [s[0] for s in steps]
    z_chain = torch.cat([step[0][None] for step in z_steps], 0)
    with torch.no_grad():
        chain = gfn(z_chain)
    plot(("{}/{:0%d}.png" % niters_digs).format(z_sgld_chain_dir, itr),
            chain.view(chain.size(0), *args.data_size))