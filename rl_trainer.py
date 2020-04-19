import torch
from torch.utils.data import DataLoader


def train(model_agent, train_set, test_set, epochs, print_epochs=1, loss_glider=20, step_glider=10, optimal_step=2,
          max_steps=100, batch_size_train=128, batch_size_test=128):

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False, drop_last=True)

    opt = torch.optim.SGD(model_agent.parameters(), lr=.01)

    print("warm-up ", end='')
    with torch.no_grad():
        model_agent.eval()

        mean_loss = 0.
        for i, data in enumerate(test_loader):
            mean_loss = (model_agent.loss(data).cpu() + mean_loss * i) / (i+1)
    print("finished")

    des_loss = (mean_loss + .01) * 2
    opt.lr = .01  # min(.01 / mean_loss, .1)
    gliding_loss = mean_loss
    gliding_step = float(optimal_step)

    model_agent.train()
    sum_steps = 0.
    sum_loss = 0.
    for epoch in range(1, epochs+1):
        epoch_loss = 0.
        epoch_steps = 0
        for i, data in enumerate(train_loader):
            loss, step = train_single(model_agent, data, opt=opt, des_loss=des_loss, console=False, max_steps=max_steps)

            gliding_loss = (gliding_loss * loss_glider + loss) / (loss_glider + 1)
            gliding_step = (gliding_step * step_glider + step) / (step_glider + 1)
            des_loss = des_loss * (1.01 if gliding_step > optimal_step else 0.99)
            opt.lr = .01  # min(.01 / gliding_loss, .1)

            epoch_steps = (epoch_steps * i + step) / (i+1)
            epoch_loss = (epoch_loss * i + loss) / (i+1)

        sum_steps += epoch_steps
        sum_loss += epoch_loss

        if epoch % print_epochs == 0:
            print("epoch: {}, des_loss: {:.3f}, mean loss: {:.4f}, mean steps: {:.4f}".format(
                epoch, des_loss, sum_loss / print_epochs, sum_steps/print_epochs
            ))
            sum_steps = 0
            sum_loss = 0

    model_agent.eval()
    evaluate(model_agent, train_loader)


def train_single(model, data, opt=None, des_loss=float('inf'), zero_step=True, console=True, max_steps=float('inf')):

    steps = 0

    loss = model.loss(data)
    first_loss = float(loss)
    if not zero_step or loss > des_loss:
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1

    while loss > des_loss and steps < max_steps:
        loss = model.loss(data)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1

    if console:
        print("loss: {:.4f}, steps: {}".format(first_loss, steps))
    return first_loss, steps


def evaluate(model, train_loader):
    sum_c1 = None
    sum_t1 = None
    sum_c0 = None
    sum_t0 = None

    mean_loss = 0.
    for i, data in enumerate(train_loader):
        correct_one, total_one, correct_zero, total_zero = model.reward_accuracy(data)
        if i == 0:
            sum_c1 = correct_one
            sum_t1 = total_one
            sum_c0 = correct_zero
            sum_t0 = total_zero
        else:
            sum_c1 += correct_one
            sum_t1 += total_one
            sum_c0 += correct_zero
            sum_t0 += total_zero
        mean_loss = (mean_loss * i + model.loss(data).cpu()) / (i + 1)
    print()
    for i in range(sum_c1.shape[0]):
        print("total: {:.3f} %, ones: {:.3f} %, zeros: {:.3f} %, dataset_split: {:.3f} %".format(
            float(sum_c1[i] + sum_c0[i])/float(sum_t1[i] + sum_t0[i])*100,
            float(sum_c1[i])/float(sum_t1[i])*100,
            float(sum_c0[i])/float(sum_t0[i])*100,
            float(sum_t1[i])/float(sum_t1[i] + sum_t0[i])*100
        ))
    print("mean loss: {:.4f}".format(mean_loss))
