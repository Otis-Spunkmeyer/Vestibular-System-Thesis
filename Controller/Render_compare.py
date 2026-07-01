import balance
from graphviz import Digraph
from sns_toolbox.color_utilities import set_text_color

def build_graph(net, engine='dot', rankdir='TB', ranksep='1.5',
                nodesep='0.8', splines='curved', size='30,30!',
                show_labels=True, clusters=None):

    graph = Digraph(engine=engine)
    graph.attr(rankdir=rankdir, ranksep=ranksep, nodesep=nodesep,
               splines=splines, size=size, overlap='false')
    graph.attr('node', fontsize='9')
    graph.attr('edge', fontsize='7')

    # build name -> index map for clustering
    name_to_idx = {pop['name']: str(i) for i, pop in enumerate(net.populations)}

    # add nodes, either clustered or flat
    if clusters:
        for cluster_name, neuron_names in clusters.items():
            with graph.subgraph(name='cluster_' + cluster_name) as c:
                c.attr(label=cluster_name, style='rounded,dashed', color='gray')
                for nname in neuron_names:
                    if nname in name_to_idx:
                        i = int(name_to_idx[nname])
                        pop = net.populations[i]
                        color_cell = pop['color']
                        color_font = set_text_color(color_cell)
                        shape = 'ellipse' if pop['number'] == 1 else 'rect'
                        style = 'filled' if pop['number'] == 1 else 'filled,rounded'
                        c.node(name_to_idx[nname], label=nname, style=style,
                               shape=shape, fillcolor=color_cell, fontcolor=color_font)
        # add any unclustered neurons
        clustered = {n for names in clusters.values() for n in names}
        for i, pop in enumerate(net.populations):
            if pop['name'] not in clustered:
                color_cell = pop['color']
                color_font = set_text_color(color_cell)
                shape = 'ellipse' if pop['number'] == 1 else 'rect'
                style = 'filled' if pop['number'] == 1 else 'filled,rounded'
                graph.node(str(i), label=pop['name'], style=style, shape=shape,
                           fillcolor=color_cell, fontcolor=color_font)
    else:
        for i, pop in enumerate(net.populations):
            color_cell = pop['color']
            color_font = set_text_color(color_cell)
            shape = 'ellipse' if pop['number'] == 1 else 'rect'
            style = 'filled' if pop['number'] == 1 else 'filled,rounded'
            graph.node(str(i), label=pop['name'], style=style, shape=shape,
                       fillcolor=color_cell, fontcolor=color_font)

    # inputs
    for i, inp in enumerate(net.inputs):
        color_cell = inp['color']
        color_font = set_text_color(color_cell)
        graph.node('In'+str(i), label=inp['name'], style='filled',
                   shape='invhouse', fillcolor=color_cell, fontcolor=color_font)
        graph.edge('In'+str(i), str(inp['destination']))

    # outputs
    for i, out in enumerate(net.outputs):
        color_cell = out['color']
        color_font = set_text_color(color_cell)
        graph.node('Out'+str(i), label=out['name'], style='filled',
                   shape='house', fillcolor=color_cell, fontcolor=color_font)
        graph.edge(str(out['source']), 'Out'+str(i))

    # connections
    for conn in net.connections:
        src = str(conn['source'])
        dst = str(conn['destination'])
        params = conn['params']
        label = conn['name'] if show_labels else ''
        if params['reversal_potential'] > 0:
            style = 'invempty'
        elif params['reversal_potential'] < 0:
            style = 'dot'
        else:
            style = 'odot'
        graph.edge(src, dst, dir='forward', arrowhead=style,
                   arrowtail=style, label=label)

    return graph


net = balance.generate_sns((1, 1, 1, 1))

clusters = {
    'Sensory Reweighting': ['01_bf_input', '02_bf_ref',
                            '33_bf_err_CCW', '34_bf_err_CW',
                            '35_bs_err_CCW', '36_bs_err_CW'],
    'Combined Error':      ['03_err_CCW', '15_err_CW'],
    'CCW Derivative':      ['04_t1', '05_t2_gt_t1',
                            '06_pos_dErr_dt', '07_neg_dErr_dt'],
    'CW Derivative':       ['17_t1', '18_t2_gt_t1',
                            '19_pos_dErr_dt', '20_neg_dErr_dt'],
    'Kp Gain':             ['08_kp', '09_prop_gain',
                            '10_kp_x_err', '16_kp_x_err'],
    'Kd Gain':             ['11_kd', '12_deriv_gain',
                            '13_kd_x_err', '21_kd_x_err'],
    'Kc Gain':             ['27_kc', '28_deriv_gain',
                            '29_kc_x_err', '30_kc_x_err'],
    'Kt / Ib Feedback':    ['23_kt', '24_int_gain',
                            '25_kt_x_t', '26_kt_x_t'],
    'Output':              ['14_PD_output', '22_PD_output'],
}

options = {
    'E_fdp_labels':   dict(engine='fdp',  rankdir='TB', size='36,36!', show_labels=True,  clusters=None),
    'F_clustered':    dict(engine='dot',  rankdir='LR', size='40,24!', show_labels=True,  clusters=clusters),
}

for name, kwargs in options.items():
    g = build_graph(net, **kwargs)
    g.format = 'png'
    g.render(filename=f'render_{name}', view=False, cleanup=True)
    print(f"Saved render_{name}.png")

