import torch, environment
env = environment.VecDrone2D(num_envs=100, device="cpu")
env.reset()

for t in range(5):
    a = torch.zeros(100, 2)              # zero action = hover thrust
    obs, r, dones, extras = env.step(a)
    x, y = env.state[:,0], env.state[:,1]
    print(f"step {t}: dones={dones.sum().item():3d}  "
          f"y=[{y.min():.2f},{y.max():.2f}]  x=[{x.min():.2f},{x.max():.2f}]")
    # which condition, for the ones that terminated:
    crashed = y <= 0.0
    oob = (x.abs() > environment.WORLD_X) | (y > environment.WORLD_Y)
    obst_d2 = (x-env.obstacle[:,0])**2 + (y-env.obstacle[:,1])**2
    hit = obst_d2 < env.OBSTACLE_RADIUS_SQ
    print(f"          crashed={crashed.sum().item()} oob={oob.sum().item()} hit={hit.sum().item()}")