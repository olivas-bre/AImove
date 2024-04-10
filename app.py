import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objs as go

from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *
from vae_gom import *
from ae_gom import *
from t_gom import *
from kf_gom import *
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, PointDrawTool, BoxSelectTool, LassoSelectTool
from io import StringIO
import param

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pn.extension('plotly', 'tabulator', template='material', raw_css=[':root { --design-primary-color: #005f9f; --design-background-color: #005f9f; --design-surface-color: #005f9f;}'])


pn.config.raw_css = [
    '.title {color: #666666 !important; font-family: "Segoe UI", sans-serif; font-weight: bold !important; font-size: large !important;}',
    '.bk-root {font-family: "Segoe UI", sans-serif;}',
    '.app-logo {height: 55px !important;}',
    '.mdc-button {color: #005f9f !important;}',
    '.mdc-icon-button.mdc-ripple-upgraded {color: #666666 !important;}',
]


offset_time = 5
dat_inPred = 0
width_A_buttons = 150


# Initialize the approaches for human motion analysis
vae_gom = VaeGom(variables, varCoef, vae_encoder, vae_decoder) # VAE-RGOM: Variational Autoencoder 
ae_gom = AeGom(variables, varCoef, modelAE) # ATT-RGOM: Autoencoder with Attention Mechanism
t_gom = TGom(variables, varCoef, model_T) # T-RGOM: Transformer
kf_gom = KfGom(variables, coef_labels_kf) # KF-GOM: Vector Autoregressive Model - Kalman Filter

#%%

# Load initial file
file = 'S4P07R1'
path='bvh2/'+file+'.bvh' # Path to the BVH file. All files from the list are in the folder 'bvh2'

# Load the BVH file with Pymo:
parser = BVHParser()
parsed_data = parser.parse(path)
mp = MocapParameterizer('position')

file_name = file+'.bvh' # Name of the file
positions = mp.fit_transform([parsed_data]) # Get the position data
jointA = variables[0] # Select the first joint angle to plot
mocap_data = parsed_data.values.reset_index().iloc[:,1:] # Get the data from the BVH file

rot_cols = [col for col in mocap_data.columns if 'rotation' in col] # Get the rotation (joint angles) columns as there are other columns with local position data
eulerAngles = mocap_data[rot_cols] # Get the joint angles data 

jpos = positions[0].values.iloc[1:,:].reset_index(drop=True) # Get the global joint positions data
joints_to_draw = list(positions[0].skeleton.keys()) # Get the list of joints to draw the stick figure

GOM_df = pd.DataFrame(index=joints_to_draw, columns=joints_to_draw) # Create a DataFrame to store the colors of the joints when dexerity analysis is selected

GOM_Skel = False # Variable to indicate if the dexterity analysis is selected

df_joints = pd.DataFrame({'Joint Angles': variables}) # Create a DataFrame to store the joint angles to select

data_marker_changes = {} # Dictionary to store the changes in the joint angles

eulerAngles_GOM = [] # List to store the joint angles for the dexterity analysis (excludes fingers and toes)
eulerAngles_original = eulerAngles.copy() # Copy of the original joint angles data
compare = False # Variable to indicate if a comparison is selected

nosynergy = ['Spine','Spine1','Spine2','Spine3','Hips','Neck','Head']  # Joints that do not have the synergy assumption

# Fill the DataFrame with colors based on the conditions
for col in GOM_df.columns:
    for idx in GOM_df.index:
        if col == idx:
            GOM_df.loc[idx, col] = 'red'  # Selected Joint
        elif positions[0].skeleton[idx]['children'] is not None and col in positions[0].skeleton[idx]['children']:
            GOM_df.loc[idx, col] = 'darkgreen'  # Serial Joint
        elif positions[0].skeleton[idx]['parent'] is not None and col in positions[0].skeleton[idx]['parent']:
            GOM_df.loc[idx, col] = 'darkgreen'  # Serial Joint
        else:
            GOM_df.loc[idx, col] = 'midnightblue'  # Non-Serial Joint
        

# Objects of the sliders that are shown to modify the constant coefficients in the KF-GOM approach
class KF_Variable(param.Parameterized):
    value = param.Number(default=0, bounds=(-20, 20), step=0.01) # Slider to modify the constant coefficient
    reset_action = param.Action(lambda x: x.param.trigger('reset'), label='Reset') # Button to reset the slider

    def __init__(self, coef=0, bounds=(-20, 20), **params): # Initialize the slider with the coefficient and bounds
        super().__init__(**params)
        self.param['value'].bounds = bounds
        self.value = coef
        self.coef = coef
        self.reset_button = pn.widgets.Button(name='Reset', width=60)  # Button to reset the slider
        self.reset_button.on_click(self._reset)  # Link the button to the reset action

    def _reset(self, event=None):
        self.value = self.coef

# Function to draw the stick figure in 3D using Plotly. Requires the obtained position data from the BVH file (positions[0]), 
# the frame to draw, the camera position, the joints to draw (not used), the option to draw the names of the joints (not used), 
# the joint to highlight (selected joint), the option to draw joints accoriding to the dexterity analysis, 
# and the option if the plot is for comparison
def draw_stickfigure3d_js(mocap_track, frame, cam=None, joints=None, draw_names=False, highlight_joint = None, GOM_Skel=False, compare = False):
    global offset_time
    frame = frame + offset_time
    layout = go.Layout(
        scene={
            'xaxis': {'range': [-800, 800]},
            'yaxis': {'range': [-800, 800]},
            'zaxis': {'range': [-800, 800]},
            'aspectmode': 'cube'
            },
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=5,
            pad=2
        ),
        autosize=False,
        width=700,
        height=500)

    fig = go.Figure(layout=layout)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    if cam is not None:
        fig.layout['scene']['camera'] = cam

    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints
    
    df = mocap_track.values

    ref_x = df['Hips_Xposition'][frame]
    ref_y = df['Hips_Zposition'][frame]
    ref_z = df['Hips_Yposition'][frame] 
    
    for joint in joints_to_draw:
        parent_x = df['%s_Xposition'%joint][frame]-ref_x
        parent_y = df['%s_Zposition'%joint][frame]-ref_y
        parent_z = df['%s_Yposition'%joint][frame]-ref_z + df['RightFootToe_Yposition'][0]
 
        # Color the joints according to the dexterity analysis if true
        if not GOM_Skel:
            fig.add_trace(go.Scatter3d(
                x=[parent_x],
                y=[parent_y],
                z=[parent_z],
                mode='markers',
                marker=dict(
                    size=6 if joint == highlight_joint else 3,
                    opacity=0.6,
                    color='red' if joint == highlight_joint else 'black'),
                marker_line=dict(width=2, color='black'),
                showlegend=False,
                text= [joint],
                hovertemplate=
                "<b>%{text}</b><br>" +
                "X: %{x:.2f}<br>" +
                "Y: %{y:.2f}<br>" +
                "Z: %{z:.2f}" +
                "<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[parent_x],
                y=[parent_y],
                z=[parent_z],
                mode='markers',
                marker=dict(
                    size=6 if joint == highlight_joint else 3,
                    opacity=0.6,
                    color=GOM_df.loc[highlight_joint, joint]), # Color the joint according to GOM_df
                marker_line=dict(width=2, color='black'),
                showlegend=False,
                text= [joint],
                hovertemplate=
                "<b>%{text}</b><br>" +
                "X: %{x:.2f}<br>" +
                "Y: %{y:.2f}<br>" +
                "Z: %{z:.2f}" +
                "<extra></extra>",
            ))

        
        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
        
        for c in children_to_draw:
            child_x = df['%s_Xposition'%c][frame]- ref_x
            child_y = df['%s_Zposition'%c][frame]- ref_y
            child_z = df['%s_Yposition'%c][frame]- ref_z + df['RightFootToe_Yposition'][0]

            if not GOM_Skel:
                fig.add_trace(go.Scatter3d(
                    x=[parent_x, child_x],
                    y=[parent_y, child_y],
                    z=[parent_z, child_z],
                    mode='lines',
                    line=dict(
                        color='black',
                        width=2),
                    hoverinfo='skip',
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[parent_x, child_x],
                    y=[parent_y, child_y],
                    z=[parent_z, child_z],
                    mode='lines',
                    line=dict(
                        color= 'rgba(0, 128, 0, 0.5)' if c == highlight_joint else GOM_df.loc[highlight_joint, c],
                        width=2),
                    hoverinfo='skip',
                    showlegend=False
                ))
            
        if draw_names:
            fig.add_trace(go.Scatter3d(
                x=[parent_x],
                y=[parent_y],
                z=[parent_z],
                mode='text',
                text=[joint],
                textposition='middle right',
                textfont=dict(
                    color='rgba(0,0,0,0.9)'),
                showlegend=False
            ))

    vertices = [
        [-200, -200, 0],
        [200, -200, 0],
        [200, 200, 0],
        [-200, 200, 0]
    ]

    # Define the faces of the grid square
    faces = [
        [0, 1, 2],
        [0, 2, 3]
    ]
    # Define the z-coordinates of the grid square
    fl = df['RightFootToe_Yposition'][0]-5 - (2*ref_z)
    z = [fl, fl, fl, fl]
    # Create a mesh object from the vertices, faces, and z-coordinates
    mesh = go.Mesh3d(
        x=[v[0] for v in vertices],
        y=[v[1] for v in vertices],
        z=z,
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        color='lightblue',
        hoverinfo='skip'
    )
    # Add the mesh object to the figure
    fig.add_trace(mesh)

    if compare:
        if frame >= len(eulerAngles_comp)-offset_time:
            frame = len(eulerAngles_comp)-offset_time-1

        pos_skel = 450
        df = positions_comp[0].values
        ref_x = df['Hips_Xposition'][frame] + pos_skel
        ref_y = df['Hips_Zposition'][frame]
        ref_z = df['Hips_Yposition'][frame] 

        for joint in joints_to_draw:
            parent_x = df['%s_Xposition'%joint][frame]-ref_x
            parent_y = df['%s_Zposition'%joint][frame]-ref_y
            parent_z = df['%s_Yposition'%joint][frame]-ref_z + df['RightFootToe_Yposition'][0]
    
            
            if not GOM_Skel:
                fig.add_trace(go.Scatter3d(
                    x=[parent_x],
                    y=[parent_y],
                    z=[parent_z],
                    mode='markers',
                    marker=dict(
                        size=6 if joint == highlight_joint else 3,
                        opacity=0.6,
                        color='red' if joint == highlight_joint else 'black'),
                    marker_line=dict(width=2, color='black'),
                    showlegend=False,
                    text= [joint],
                    hovertemplate=
                    "<b>%{text}</b><br>" +
                    "X: %{x:.2f}<br>" +
                    "Y: %{y:.2f}<br>" +
                    "Z: %{z:.2f}" +
                    "<extra></extra>",
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[parent_x],
                    y=[parent_y],
                    z=[parent_z],
                    mode='markers',
                    marker=dict(
                        size=6 if joint == highlight_joint else 3,
                        opacity=0.6,
                        color=GOM_df.loc[highlight_joint, joint]),
                    marker_line=dict(width=2, color='black'),
                    showlegend=False,
                    text= [joint],
                    hovertemplate=
                    "<b>%{text}</b><br>" +
                    "X: %{x:.2f}<br>" +
                    "Y: %{y:.2f}<br>" +
                    "Z: %{z:.2f}" +
                    "<extra></extra>",
                ))

            
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
            
            for c in children_to_draw:
                child_x = df['%s_Xposition'%c][frame]- ref_x
                child_y = df['%s_Zposition'%c][frame]- ref_y
                child_z = df['%s_Yposition'%c][frame]- ref_z + df['RightFootToe_Yposition'][0]

                if not GOM_Skel:
                    fig.add_trace(go.Scatter3d(
                        x=[parent_x, child_x],
                        y=[parent_y, child_y],
                        z=[parent_z, child_z],
                        mode='lines',
                        line=dict(
                            color='black',
                            width=2),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[parent_x, child_x],
                        y=[parent_y, child_y],
                        z=[parent_z, child_z],
                        mode='lines',
                        line=dict(
                            color= 'rgba(0, 128, 0, 0.5)' if c == highlight_joint else GOM_df.loc[highlight_joint, c],
                            width=2),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
            if draw_names:
                fig.add_trace(go.Scatter3d(
                    x=[parent_x],
                    y=[parent_y],
                    z=[parent_z],
                    mode='text',
                    text=[joint],
                    textposition='middle right',
                    textfont=dict(
                        color='rgba(0,0,0,0.9)'),
                    showlegend=False
                ))

        vertices = [
            [-650, -200, 0],
            [-250, -200, 0],
            [-250, 200, 0],
            [-650, 200, 0]
        ]

        # Define the faces of the grid square
        faces = [
            [0, 1, 2],
            [0, 2, 3]
        ]
        # Define the z-coordinates of the grid square
        fl = df['RightFootToe_Yposition'][0]-5 - (2*ref_z)
        z = [fl, fl, fl, fl]
        # Create a mesh object from the vertices, faces, and z-coordinates
        mesh = go.Mesh3d(
            x=[v[0] for v in vertices],
            y=[v[1] for v in vertices],
            z=z,
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            color='lightgreen',
            hoverinfo='skip'
        )
        # Add the mesh object to the figure
        fig.add_trace(mesh)

    return fig


#%%

# Variables to store the loaded files and models, to avoid loading them again
file_loaded = ''
model_loaded = ''

# Initial code to show the animation card and have ready the plots for the joint angles
i_angle = joints_to_draw[0]
dff_3d = eulerAngles[[i_angle+'_Xrotation', i_angle+'_Yrotation', i_angle+'_Zrotation']]

# 2D plots
fig2Dx = go.Figure([
    go.Scatter(
        name=dff_3d.columns[0],
        x=np.arange(0, len(dff_3d.iloc[offset_time:,0])),
        y=dff_3d.iloc[offset_time:,0],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    )], layout=dict(width=690, height=450))

fig2Dx.update_xaxes(title_text='Time frame')
fig2Dx.update_yaxes(title_text='Angle (degrees)')


fig2Dy = go.Figure([
    go.Scatter(
        name=dff_3d.columns[1],
        x=np.arange(0, len(dff_3d.iloc[offset_time:,1])),
        y=dff_3d.iloc[offset_time:,1],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    )], layout=dict(width=690, height=450))

fig2Dy.update_xaxes(title_text='Time frame')
fig2Dy.update_yaxes(title_text='Angle (degrees)')

fig2Dz = go.Figure([
    go.Scatter(
        name=dff_3d.columns[2],
        x=np.arange(0, len(dff_3d.iloc[offset_time:,2])),
        y=dff_3d.iloc[offset_time:,2],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    )], layout=dict(width=690, height=450))


fig2Dz.update_xaxes(title_text='Time frame')
fig2Dz.update_yaxes(title_text='Angle (degrees)')


# 3D plot
fig3D = go.Figure(data=go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], mode='lines'),
                  layout=dict(width=690, height=450))
fig3D.add_trace(go.Scatter3d(x=[dff_3d.iloc[offset_time,0]], y=[dff_3d.iloc[offset_time,1]], z=[dff_3d.iloc[offset_time,2]], mode='markers',
                             showlegend=False, marker=dict(size=5, symbol='diamond')))
figA = draw_stickfigure3d_js(positions[0], frame=0, highlight_joint = joints_to_draw[0], GOM_Skel=GOM_Skel)

fig2Dx_pane = pn.pane.Plotly(fig2Dx)
fig2Dy_pane = pn.pane.Plotly(fig2Dy)
fig2Dz_pane = pn.pane.Plotly(fig2Dz)
fig3D_pane = pn.pane.Plotly(fig3D)
figA_pane = pn.pane.Plotly(figA)

tabs = pn.Tabs(('X', fig2Dx_pane), ('Y', fig2Dy_pane),('Z', fig2Dz_pane))


#%% Create the main widgets for the app
uploadFile = pn.widgets.FileInput(accept='.bvh', name='Upload a new BVH file') # Upload a new BVH file
selectFile = pn.widgets.Select(name='Select file', 
                               groups={'ERGD': ['S4P07R1', 'S4P07R2'], 'Silk weaving': ['SWMLS01G01R01', 'SWMLS01G01R02'], 
                                       'Glassblowing': ['GBBSS01G03R01', 'GBBSS01G03R02'], 'Mastic cultivation': ['MCEAS02G01R01', 'MCEAS02G01R02'],
                                       'Television Assembly': ['TVBS01P01R02', 'TVBS01P02R07', 'TVBS01P03R09', 'TV_S01P01R13'],
                                       'Airplane Assembly': ['PLNS01P02R05', 'PLNS02P03R02'],
                                       'Ganterie': ['Ganterie_example_1', 'Ganterie_example_2']}, value = 'S4P07R1') # Select the file to load from ready files

button_file = pn.widgets.Button(name='Load MoCap file', button_type='primary') # Button to load the selected file

selectAngle = pn.widgets.Select(name='Select Joint to Plot', options= joints_to_draw, value = joints_to_draw[0]) # Select the joint to plot
selectAngleCoef = pn.widgets.Select(name='Select Joint Angle Axis:', options= ['X', 'Y', 'Z'], value = 'X') # Select the axis to plot the joint angle

frame_slider = pn.widgets.EditableIntSlider(name='Frame', start=1, end=len(jpos)-offset_time, step=1, value=0) # Slider to select the frame to show 
selectInterval = pn.widgets.Select(name='Select Interval', options= ['None', 'Confidence intervals'], value = 'None') # Select the intervals to show in the plots

checkbox2D = pn.widgets.Checkbox(name='Show 2D Joint Angle Trajectories') # Shows the 2D plot
checkbox3D = pn.widgets.Checkbox(name='Show 3D Joint Angle Trajectories') # Shows the 3D plot
checkboxGOM = pn.widgets.Checkbox(name='Dexterity Analysis') # Shows the GOM card
checkboxKinematics = pn.widgets.Checkbox(name='Kinematics') #Not implemented yet

# This sliders is to indicate up to which frames to show the intervals, so not all are computed, only the ones close to the marker (current frame)
int_range_slider = pn.widgets.IntRangeSlider(name='Interval Range',
    start=-10, end=10, value=(0, 0), step=1)


selectCompareFile = pn.widgets.Select(name='Select an Existing File', 
                               groups={'ERGD': ['S4P07R1', 'S4P07R2'], 'Silk weaving': ['SWMLS01G01R01', 'SWMLS01G01R02'], 
                                       'Glassblowing': ['GBBSS01G03R01', 'GBBSS01G03R02'], 'Mastic cultivation': ['MCEAS02G01R01', 'MCEAS02G01R02'],
                                       'Television Assembly': ['TVBS01P01R02', 'TVBS01P02R07', 'TVBS01P03R09', 'TV_S01P01R13'],
                                       'Airplane Assembly': ['PLNS01P02R05', 'PLNS02P03R02'],
                                       'Ganterie': ['Ganterie_example_1', 'Ganterie_example_2']}, value = 'S4P07R1') # Select the file to compare

inputCompareFile = pn.widgets.FileInput(accept='.bvh', name='Upload a new BVH file') # Upload a new BVH file to compare

buttonRemoveCompare = pn.widgets.Button(name='Remove comparison', button_type='warning') # Button to remove the comparison
buttonCompare = pn.widgets.Button(name='Compare', button_type='primary') # Button to compare the selected file

# Create the widgets for the GOM card
# This function is so only one toggle button can be selected at a time
def sync_toggle_groups(event):
    if event.new is not None:
        if event.obj is toggle_Model1:
            toggle_Model2.value = None
        elif event.obj is toggle_Model2:
            toggle_Model1.value = None

# Create the toggle buttons for the different data-intesive models
toggle_Model1 = pn.widgets.ToggleGroup(name='Data-intensive', options=['VAE-RGOM', 'ATT-RGOM', 'T-RGOM'], behavior="radio", value='ATT-RGOM', label='Data-intensive')
toggle_Model1.param.watch(sync_toggle_groups, 'value') # Link the toggle buttons to the function

# Create the toggle button for the one-shot model
toggle_Model2 = pn.widgets.ToggleGroup(name='One-Shot', options=['KF-GOM'], behavior="radio", value=None, width=100, label='One-Shot')
toggle_Model2.param.watch(sync_toggle_groups, 'value') # Link the toggle button to the function

# Create the static text to show the selected model
toggle_Model = pn.widgets.StaticText(value=toggle_Model1.value or toggle_Model2.value)

# Function to update the selected model
def update_selected_option(event):
    if event.new is not None:
        #print('Selected model:', event.new)
        toggle_Model.value = event.new

toggle_Model1.param.watch(update_selected_option, 'value')
toggle_Model2.param.watch(update_selected_option, 'value')

toggle_Model2.value = None

# Visualization of the widgets on the GOM card
toggle_Model_Widgets = pn.Row(
    pn.Column(pn.pane.Markdown('### Data-intensive'), toggle_Model1, background='whitesmoke', width=320),
    pn.Spacer(width=50),
    pn.Column(pn.pane.Markdown('### One-Shot'), pn.Column(toggle_Model2, align='center'), background='whitesmoke', width=320))


#%% Create the markers for the initial angle of the selected joint that is shown in the 2D plot (GOM card)
# The markers are created every 45 frames, which can be modified by the user in the app
initial_angle = eulerAngles[variables[0]].iloc[offset_time:]
n = len(initial_angle.index) // 45
x_marker = initial_angle.index[::n]
y_marker = initial_angle.values[::n]

x_marker = np.concatenate(([initial_angle.index[0]], x_marker, [initial_angle.index[-1]]))
y_marker = np.concatenate(([initial_angle.values[0]], y_marker, [initial_angle.values[-1]]))

data_marker = ColumnDataSource({
    'x': x_marker,
    'y': y_marker,
})

p = figure(title=variables[0], x_axis_label='Frames', y_axis_label='Angle', plot_height=400)
source = ColumnDataSource(data=dict(x=initial_angle.index, y=initial_angle.values))
p.line('x', 'y', source=source, line_width=2, color='#D8BFD8')
renderer = p.scatter(x='x', y='y', source=data_marker, color='#800080', size=10)
p.line(x='x', y='y', source=data_marker, color='#800080')

draw_tool = PointDrawTool(renderers=[renderer], empty_value='black')
p.add_tools(draw_tool)
box_select = BoxSelectTool(renderers=[renderer])
lasso_select = LassoSelectTool(renderers=[renderer])
p.add_tools(box_select, lasso_select)
plot_pane = pn.pane.Bokeh(p)


# Create the markers for the comparison plot, to have ready for the comparison of the joint angles
source_comp = ColumnDataSource(data=dict(x=[], y=[]))
p.line('x', 'y', source=source_comp, line_width=2, color='black', line_dash='dashed')


# Create all my widgets for modifying the joint angles
checkbox_group = pn.widgets.Tabulator(df_joints, selection=list(df_joints.index), height=400, selectable='checkbox') # Table to select the joint angles to modify
save_button = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons) # Button to save the changes
reset_Jbutton = pn.widgets.Button(name="Reset Angle Changes", button_type="warning", width=width_A_buttons) # Button to reset the changes
reset_Abutton = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons) # Button to reset all the changes
accept_button = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons) # Button to accept the changes

#%% Function to update the 2D plots with the selected joint angle

def update_angle(selectAngle):
    global offset_time, eulerAngles
    #print('Change Angle: ', selectAngle)
    dff_3d = eulerAngles[[selectAngle+'_Xrotation', selectAngle+'_Yrotation', selectAngle+'_Zrotation']]

    fig3D = go.Figure(layout=dict(width=690, height=450))
    fig2Dx = go.Figure(layout=dict(width=690, height=450))
    fig2Dy = go.Figure(layout=dict(width=690, height=450))
    fig2Dz= go.Figure(layout=dict(width=690, height=450))

    fig2Dx.add_trace(
        go.Scatter(
            name=dff_3d.columns[0],
            x=np.arange(0, len(dff_3d.iloc[offset_time:,0])),
            y=dff_3d.iloc[offset_time:,0],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dx.add_trace(
        go.Scatter(
            x=[0],  
            y=[dff_3d.iloc[offset_time,0]],  
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
    ))
    fig2Dx.update_xaxes(title_text='Time frame')
    fig2Dx.update_yaxes(title_text='Angle (degrees)')

    fig2Dy.add_trace(
        go.Scatter(
            name=dff_3d.columns[1],
            x=np.arange(0, len(dff_3d.iloc[offset_time:,1])),
            y=dff_3d.iloc[offset_time:,1],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dy.add_trace(
        go.Scatter(
            x=[0],  
            y=[dff_3d.iloc[offset_time,1]],  
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
    ))
    fig2Dy.update_xaxes(title_text='Time frame')
    fig2Dy.update_yaxes(title_text='Angle (degrees)')

    fig2Dz.add_trace(
        go.Scatter(
            name=dff_3d.columns[2],
            x=np.arange(0, len(dff_3d.iloc[offset_time:,2])),
            y=dff_3d.iloc[offset_time:,2],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dz.add_trace(
        go.Scatter(
            x=[0],  
            y=[dff_3d.iloc[offset_time,2]],  
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
    ))
    fig2Dz.update_xaxes(title_text='Time frame')
    fig2Dz.update_yaxes(title_text='Angle (degrees)')
    
    fig3D.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], 
                                 mode='lines', name=selectAngle))
    fig3D.add_trace(go.Scatter3d(x=[dff_3d.iloc[offset_time,0]], y=[dff_3d.iloc[offset_time,1]], z=[dff_3d.iloc[offset_time,2]], 
                                 mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))
    
    # Add the joint angle of the recording file that is compated
    if compare == True:
        dff_3d = eulerAngles_comp[[selectAngle+'_Xrotation', selectAngle+'_Yrotation', selectAngle+'_Zrotation']]

        fig2Dx.add_trace(
            go.Scatter(
                name=dff_3d.columns[0]+' (2nd recording)',
                x=np.arange(0, len(dff_3d.iloc[offset_time:,0])),
                y=dff_3d.iloc[offset_time:,0],
                mode='lines'
            ))
        fig2Dx.add_trace(
            go.Scatter(
                x=[0],  
                y=[dff_3d.iloc[offset_time,0]],  
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
        ))


        fig2Dy.add_trace(
            go.Scatter(
                name=dff_3d.columns[1]+' (2nd recording)',
                x=np.arange(0, len(dff_3d.iloc[offset_time:,1])),
                y=dff_3d.iloc[offset_time:,1],
                mode='lines'
            ))
        fig2Dy.add_trace(
            go.Scatter(
                x=[0],  
                y=[dff_3d.iloc[offset_time,1]],  
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
        ))

        fig2Dz.add_trace(
            go.Scatter(
                name=dff_3d.columns[2]+' (2nd recording)',
                x=np.arange(0, len(dff_3d.iloc[offset_time:,2])),
                y=dff_3d.iloc[offset_time:,2],
                mode='lines'
            ))
        fig2Dz.add_trace(
            go.Scatter(
                x=[0],  
                y=[dff_3d.iloc[offset_time,2]],  
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
        ))
        
        fig3D.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], mode='lines', 
                                    name=selectAngle+' (2nd recording)'))
        fig3D.add_trace(go.Scatter3d(x=[dff_3d.iloc[offset_time,0]], y=[dff_3d.iloc[offset_time,1]], 
                                    z=[dff_3d.iloc[offset_time,2]], mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))
    
    fig2Dx_pane.object = fig2Dx
    fig2Dx_pane.param.trigger('object')
    fig2Dy_pane.object = fig2Dy
    fig2Dy_pane.param.trigger('object')
    fig2Dz_pane.object = fig2Dz
    fig2Dz_pane.param.trigger('object')
    fig3D_pane.object = fig3D
    fig3D_pane.param.trigger('object')
    selectInterval.value = 'None'
    selectInterval.param.trigger('value')
    

iupdate_angle= pn.bind(update_angle, selectAngle)

# Function to show the selected interval range, and marker of the current frame
def update_interval(selectInterval, frame, int_frame):
    global offset_time, eulerAngles, positions, GOM_Skel
    #print('Change: ', selectInterval)
    #print('Change: ', frame)
    #print('Range: ', int_frame)

    fig3D = go.Figure(layout=dict(width=690, height=450))
    fig2Dx = go.Figure(layout=dict(width=690, height=450))
    fig2Dy = go.Figure(layout=dict(width=690, height=450))
    fig2Dz= go.Figure(layout=dict(width=690, height=450))

    dff_3d = eulerAngles[[selectAngle.value+'_Xrotation', selectAngle.value+'_Yrotation', selectAngle.value+'_Zrotation']]

    dat_x = dff_3d.iloc[offset_time:,0].values
    dat_y = dff_3d.iloc[offset_time:,1].values
    dat_z = dff_3d.iloc[offset_time:,2].values   

    fig2Dx.add_trace(
        go.Scatter(
            name=dff_3d.columns[0],
            x=np.arange(0, len(dat_x)),
            y=dat_x,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dx.add_trace(
        go.Scatter(
            x=[frame],  
            y=[dat_x[frame]], # Marker at the current frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
    ))
    fig2Dx.update_xaxes(title_text='Time frame')
    fig2Dx.update_yaxes(title_text='Angle (degrees)')

    fig2Dy.add_trace(
        go.Scatter(
            name=dff_3d.columns[1],
            x=np.arange(0, len(dat_y)),
            y=dat_y,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dy.add_trace(
        go.Scatter(
            x=[frame], 
            y=[dat_y[frame]],  
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
    ))
    fig2Dy.update_xaxes(title_text='Time frame')
    fig2Dy.update_yaxes(title_text='Angle (degrees)')

    fig2Dz.add_trace(
        go.Scatter(
            name=dff_3d.columns[2],
            x=np.arange(0, len(dat_z)),
            y=dat_z,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dz.add_trace(
        go.Scatter(
            x=[frame],  
            y=[dat_z[frame]],  
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
    ))
    fig2Dz.update_xaxes(title_text='Time frame')
    fig2Dz.update_yaxes(title_text='Angle (degrees)')


    cam3D = fig3D_pane.object.layout['scene']['camera']
    fig3D.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], 
                                 mode='lines', name=selectAngle.value))
    fig3D.add_trace(go.Scatter3d(x=[dff_3d.iloc[frame,0]], y=[dff_3d.iloc[frame,1]], z=[dff_3d.iloc[frame,2]], 
                                 mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))
    
    if selectInterval == 'Confidence intervals':
        alpha = 0.05
        ci_x = (1-(alpha/2))*np.std(dat_x)/np.mean(dat_x)
        ci_y = (1-(alpha/2))*np.std(dat_y)/np.mean(dat_y)
        ci_z = (1-(alpha/2))*np.std(dat_z)/np.mean(dat_z)
        
        fig2Dx.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=np.arange(0, len(dat_x)),
                y=dat_x+ci_x,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))
        fig2Dx.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=np.arange(0, len(dat_x)),
                y=dat_x-ci_x,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

        fig2Dy.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=np.arange(0, len(dat_y)),
                y=dat_y+ci_y,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))
        fig2Dy.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=np.arange(0, len(dat_y)),
                y=dat_y-ci_y,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

        fig2Dz.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=np.arange(0, len(dat_z)),
                y=dat_z+ci_z,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))
        fig2Dz.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=np.arange(0, len(dat_z)),
                y=dat_z-ci_z,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

        # Add the confidence intervals to the 3D plot 
        # Ellipsoids are put here, but correct intervals should be implemented, confidence intervals are not good option for this
        # Other intervals need to be implemented that are easy to calculate and show in the 3D plot
        radii = [ci_x, ci_y, ci_z]  

        n = len(jpos)-offset_time

        low_frame = frame + int_frame[0]
        up_frame = frame + int_frame[1]

        if low_frame <= 0:
            start, end = 0, up_frame
        elif up_frame >= n:
            start, end = low_frame, n
        else:
            start, end = low_frame, up_frame

        for tt in range(start, end):
            center = [dat_x[tt], dat_y[tt], dat_z[tt]]
            phi = np.linspace(0, 2*np.pi, 30)
            theta = np.linspace(0, np.pi, 30)
            x = radii[0] * np.outer(np.cos(phi), np.sin(theta)) + center[0]
            y = radii[1] * np.outer(np.sin(phi), np.sin(theta)) + center[1]
            z = radii[2] * np.outer(np.ones(np.size(phi)), np.cos(theta)) + center[2]
            fig3D.add_trace(go.Mesh3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), opacity=0.2, color='rgba(200, 100, 200, 0.5)'))
    
    # Add the joint angle of the recording file that is compated
    if compare == True:
        if frame >= len(eulerAngles_comp)-offset_time:
            frame = len(eulerAngles_comp)-offset_time-1

        dff_3d = eulerAngles_comp[[selectAngle.value+'_Xrotation', selectAngle.value+'_Yrotation', selectAngle.value+'_Zrotation']]

        dat_x = dff_3d.iloc[offset_time:,0].values
        dat_y = dff_3d.iloc[offset_time:,1].values
        dat_z = dff_3d.iloc[offset_time:,2].values   

        fig2Dx.add_trace(
            go.Scatter(
                name=dff_3d.columns[0] + ' (2nd recording)',
                x=np.arange(0, len(dat_x)),
                y=dat_x,
                mode='lines'
            ))
        fig2Dx.add_trace(
            go.Scatter(
                x=[frame],  # Replace with the x-coordinate of the marker in the initial frame
                y=[dat_x[frame]],  # Replace with the y-coordinate of the marker in the initial frame
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
        ))

        fig2Dy.add_trace(
            go.Scatter(
                name=dff_3d.columns[1]+' (2nd recording)',
                x=np.arange(0, len(dat_y)),
                y=dat_y,
                mode='lines'
            ))
        fig2Dy.add_trace(
            go.Scatter(
                x=[frame],  # Replace with the x-coordinate of the marker in the initial frame
                y=[dat_y[frame]],  # Replace with the y-coordinate of the marker in the initial frame
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
        ))

        fig2Dz.add_trace(
            go.Scatter(
                name=dff_3d.columns[2]+' (2nd recording)',
                x=np.arange(0, len(dat_z)),
                y=dat_z,
                mode='lines'
            ))
        fig2Dz.add_trace(
            go.Scatter(
                x=[frame],  # Replace with the x-coordinate of the marker in the initial frame
                y=[dat_z[frame]],  # Replace with the y-coordinate of the marker in the initial frame
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
        ))
        
        fig3D.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], mode='lines', 
                                    name=selectAngle.value+' (2nd recording)'))
        fig3D.add_trace(go.Scatter3d(x=[dff_3d.iloc[frame,0]], y=[dff_3d.iloc[frame,1]], 
                                    z=[dff_3d.iloc[frame,2]], mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))
        
    fig2Dx_pane.object = fig2Dx
    fig2Dy_pane.object = fig2Dy
    fig2Dz_pane.object = fig2Dz

    fig3D.layout['scene']['camera'] = cam3D
    fig3D_pane.object = fig3D
    camA = figA_pane.object.layout['scene']['camera']
    figA_pane.object = draw_stickfigure3d_js(positions[0], frame=frame, cam=camA, highlight_joint = selectAngle.value, GOM_Skel=GOM_Skel, compare=compare)


iupdate_interval= pn.bind(update_interval, selectInterval, frame_slider, int_range_slider)

# Function to show range slider for the confidence intervals
def show_intervSlider(select_value):
    if select_value == 'Confidence intervals':
        return int_range_slider
    else:
        return None

# Function when a new file is loaded
def load_NewFile(event):
    global eulerAngles, jpos, positions, i_angle, GOM_Skel, eulerAngles_original, GOM_df, joints_to_draw
    
    checkboxGOM.value = False # Uncheck the GOM card
    
    # Load the new file and prepare all required data
    positions = mp.fit_transform([parsed_data])
    mocap_data = parsed_data.values.reset_index().iloc[:,1:]
    rot_cols = [col for col in mocap_data.columns if 'rotation' in col]
    eulerAngles = mocap_data[rot_cols]
    eulerAngles_original = eulerAngles.copy()

    jpos = positions[0].values.iloc[1:,:].reset_index(drop=True)
    joints_to_draw = list(positions[0].skeleton.keys())
    GOM_df = pd.DataFrame(index=joints_to_draw, columns=joints_to_draw)
    
    for col in GOM_df.columns:
        for idx in GOM_df.index:
            if col == idx:
                GOM_df.loc[idx, col] = 'red'  # Selected Joint
            elif positions[0].skeleton[idx]['children'] is not None and col in positions[0].skeleton[idx]['children']:
                GOM_df.loc[idx, col] = 'darkgreen'  # Serial Joint
            elif positions[0].skeleton[idx]['parent'] is not None and col in positions[0].skeleton[idx]['parent']:
                GOM_df.loc[idx, col] = 'darkgreen'  # Serial Joint
            else:
                GOM_df.loc[idx, col] = 'midnightblue'  # Non-Serial Joint
    
    frame_slider.value = 1
    frame_slider.start = 1
    frame_slider.end = len(jpos) - offset_time
    frame_slider.param.trigger('value')

    selectInterval.value = 'None'
    selectInterval.param.trigger('value')
    

    selectAngle.value = i_angle
    selectInterval.param.trigger('value')


    fig3D_LN = go.Figure(layout=dict(width=690, height=450))
    fig2Dx_LN = go.Figure(layout=dict(width=690, height=450))
    fig2Dy_LN = go.Figure(layout=dict(width=690, height=450))
    fig2Dz_LN= go.Figure(layout=dict(width=690, height=450))

    # Update the 2D and 3D plots with the new data
    dff_3d = eulerAngles[[selectAngle.value+'_Xrotation', selectAngle.value+'_Yrotation', selectAngle.value+'_Zrotation']]
    
    fig2Dx_LN.add_trace(
        go.Scatter(
            name=dff_3d.columns[0],
            x=np.arange(0, len(dff_3d.iloc[offset_time:,0])),
            y=dff_3d.iloc[offset_time:,0],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dx_LN.add_trace(
        go.Scatter(
            x=[0],  
            y=[dff_3d.iloc[offset_time,0]],  
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
    ))
    fig2Dx_LN.update_xaxes(title_text='Time frame')
    fig2Dx_LN.update_yaxes(title_text='Angle (degrees)')


    fig2Dy_LN.add_trace(
        go.Scatter(
            name=dff_3d.columns[1],
            x=np.arange(0, len(dff_3d.iloc[offset_time:,1])),
            y=dff_3d.iloc[offset_time:,1],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dy_LN.add_trace(
        go.Scatter(
            x=[0],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dff_3d.iloc[offset_time,1]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
    ))
    fig2Dy_LN.update_xaxes(title_text='Time frame')
    fig2Dy_LN.update_yaxes(title_text='Angle (degrees)')

    fig2Dz_LN.add_trace(
        go.Scatter(
            name=dff_3d.columns[2],
            x=np.arange(0, len(dff_3d.iloc[offset_time:,2])),
            y=dff_3d.iloc[offset_time:,2],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dz_LN.add_trace(
        go.Scatter(
            x=[0],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dff_3d.iloc[offset_time,2]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
    ))
    fig2Dz_LN.update_xaxes(title_text='Time frame')
    fig2Dz_LN.update_yaxes(title_text='Angle (degrees)')
    
    fig3D_LN.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], 
                                    mode='lines', name=selectAngle.value))
    fig3D_LN.add_trace(go.Scatter3d(x=[dff_3d.iloc[offset_time,0]], y=[dff_3d.iloc[offset_time,1]], z=[dff_3d.iloc[offset_time,2]], 
                                    mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))
    
    # Add the joint angle of the recording file that is compated
    if compare == True:
        dff_3d = eulerAngles_comp[[selectAngle.value+'_Xrotation', selectAngle.value+'_Yrotation', selectAngle.value+'_Zrotation']]

        fig2Dx_LN.add_trace(
            go.Scatter(
                name=dff_3d.columns[0]+' (2nd recording)',
                x=np.arange(0, len(dff_3d.iloc[offset_time:,0])),
                y=dff_3d.iloc[offset_time:,0],
                mode='lines'
            ))
        fig2Dx_LN.add_trace(
            go.Scatter(
                x=[0],  
                y=[dff_3d.iloc[offset_time,0]],  
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
        ))

        fig2Dy_LN.add_trace(
            go.Scatter(
                name=dff_3d.columns[1]+' (2nd Recording)',
                x=np.arange(0, len(dff_3d.iloc[offset_time:,1])),
                y=dff_3d.iloc[offset_time:,1],
                mode='lines'
            ))
        fig2Dy_LN.add_trace(
            go.Scatter(
                x=[0],  # Replace with the x-coordinate of the marker in the initial frame
                y=[dff_3d.iloc[offset_time,1]],  # Replace with the y-coordinate of the marker in the initial frame
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
        ))

        fig2Dz_LN.add_trace(
            go.Scatter(
                name=dff_3d.columns[2]+' (2nd Recording)',
                x=np.arange(0, len(dff_3d.iloc[offset_time:,2])),
                y=dff_3d.iloc[offset_time:,2],
                mode='lines'
            ))
        fig2Dz_LN.add_trace(
            go.Scatter(
                x=[0],  # Replace with the x-coordinate of the marker in the initial frame
                y=[dff_3d.iloc[offset_time,2]],  # Replace with the y-coordinate of the marker in the initial frame
                mode='markers',
                showlegend=False,
                marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
        ))
        
        fig3D_LN.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], mode='lines', 
                                    name=selectAngle.value+' (2nd Recording)'))
        fig3D_LN.add_trace(go.Scatter3d(x=[dff_3d.iloc[offset_time,0]], y=[dff_3d.iloc[offset_time,1]], 
                                    z=[dff_3d.iloc[offset_time,2]], mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))

    fig2Dx_pane.object = fig2Dx_LN
    fig2Dx_pane.param.trigger('object')
    fig2Dy_pane.object = fig2Dy_LN
    fig2Dy_pane.param.trigger('object')
    fig2Dz_pane.object = fig2Dz_LN
    fig2Dz_pane.param.trigger('object')
    fig3D_pane.object = fig3D_LN
    fig3D_pane.param.trigger('object')
    figA_pane.object = draw_stickfigure3d_js(positions[0], frame=0, highlight_joint = selectAngle.value, GOM_Skel=GOM_Skel, compare=compare)
    figA_pane.param.trigger('object')

    # I removed the following code because the GOM card is not shown when a new file is loaded but the dexterity checkbox is checked 
    # Need to correct
    
    # if checkboxGOM.value:
    #     global coef, pred, data_marker_changes, eulerAngles_GOM
        
        
    #     data_marker_changes = {}
    #     eulerAngles_GOM = []
    #     #coef, pred = ae_gom.do_gom(eulerAngles)
    #     if toggle_Model.value == 'ATT-RGOM':
    #         coef, pred = ae_gom.do_gom(eulerAngles)
    #         #print('ATT-RGOM')
    #     elif toggle_Model.value == 'VAE-RGOM':
    #         coef, pred = vae_gom.do_gom(eulerAngles)
    #         #print('VAE-RGOM')
    #     elif toggle_Model.value == 'T-RGOM': 
    #         coef, pred = t_gom.do_gom(eulerAngles)
    #         #print('T-RGOM')
    #     elif toggle_Model.value == 'KF-GOM': 
    #         coef, pred = kf_gom.do_gom(eulerAngles)
    #         #print('KF-GOM')
    #     selectAngleCoef.value = 'X'
    #     selectAngleCoef.param.trigger('value')

        

# Function to show the GOM card when the dexterity analysis checkbox is checked
def plot_iGOM(selectAngle, angMod, trained_model):
    global file_loaded, model_loaded, coef, pred, stats_figT, ae_gom
    global j_var, col_coef, offsetP, tabsC, col_stats, positionsPred, figA_pred_pane, download_Gdat_button
    global dfCoef_A1, angleA1, dfCoef_A1_original, data_markerA1, pA1, sourceA1, data_marker_changesA1, checkbox_groupA1
    global dfCoef_A2, angleA2, dfCoef_A2_original, data_markerA2, pA2, sourceA2, data_marker_changesA2, checkbox_groupA2
    global dfCoef_A3, angleA3, dfCoef_A3_original, data_markerA3, pA3, sourceA3, data_marker_changesA3, checkbox_groupA3, coef_A3_GOM
    global dfCoef_A41, angleA41, dfCoef_A41_original, data_markerA41, pA41, sourceA41, data_marker_changesA41, checkbox_groupA41
    global dfCoef_A42, angleA42, dfCoef_A42_original, data_markerA42, pA42, sourceA42, data_marker_changesA42, checkbox_groupA42
    global plot_paneA1, plot_paneA2, plot_paneA3, plot_paneA41, plot_paneA42, fig_pred_pane, fig_pred_plot
    global kf_variables_A1, kf_variables_A2, kf_variables_A3, kf_variables_A41, kf_variables_A42, row_coef, dfPvalues

    # Compute the prediction and coefficients according to the approach selected (I make sure to not recompute when the same file is loaded)
    if file_loaded != file_name:
        file_loaded = file_name
        if trained_model == 'ATT-RGOM':
            coef, pred = ae_gom.do_gom(eulerAngles)
            #print('ATT-RGOM')
        elif trained_model == 'VAE-RGOM':
            coef, pred = vae_gom.do_gom(eulerAngles)
            #print('VAE-RGOM')
        elif trained_model == 'T-RGOM': 
            coef, pred = t_gom.do_gom(eulerAngles)
            #print('T-RGOM')
        elif trained_model == 'KF-GOM': 
            coef, pred = kf_gom.do_gom(eulerAngles)
            #print('KF-GOM')
        #print(file_loaded)
    elif model_loaded != trained_model:
        model_loaded = trained_model
        if trained_model == 'ATT-RGOM':
            coef, pred = ae_gom.do_gom(eulerAngles)
            #print('ATT-RGOM')
        elif trained_model == 'VAE-RGOM':
            coef, pred = vae_gom.do_gom(eulerAngles)
            #print('VAE-RGOM')
        elif trained_model == 'T-RGOM': 
            coef, pred = t_gom.do_gom(eulerAngles)
            #print('T-RGOM')
        elif trained_model == 'KF-GOM': 
            coef, pred = kf_gom.do_gom(eulerAngles)
            #print('KF-GOM')

    index = variables.index(selectAngle+'_'+angMod+'rotation') #Select joint angle modeled

    if trained_model != 'KF-GOM':
        j_var = varSub[index]
        dfCoef = coef.filter(like=j_var).reset_index(drop=True)

        columns_t3 = dfCoef.filter(like='t-3').columns
        dfCoef = dfCoef.drop(columns=columns_t3)
        col_coef = dfCoef.columns.tolist()
        tab_stats = dfCoef.describe().T
        tab_stats = tab_stats.reset_index()
        tab_stats = tab_stats.rename(columns={'index': 'Joint Angle'})
        stats_figT = pn.widgets.Tabulator(tab_stats, height=300)
        model_title = pn.Row(pn.pane.Markdown('### '+file_name+': '+selectAngle+' '+angMod+'-axis'))
    else:
        dfCoef = coef.iloc[:,index]
        dfPvalues = kf_gom.pvalues.iloc[:,index]
        col_coef = coef.columns.tolist()
        row_coef = coef.index.tolist()
        tab_stats = pd.concat([dfCoef, dfPvalues], axis=1)
        tab_stats.columns = ['Coefficients', 'P-values']
        stats_figT = pn.widgets.Tabulator(tab_stats, height=300)
        model_title = pn.Row(pn.pane.Markdown('### '+file_name+': '+selectAngle+' '+angMod+'-axis'))

    # Global Statistics
    # Only with KF-GOM the p-values are shown (the p-values are not shown for the other models as require all their iterations to compute)
    col_stats = pn.Column(
        model_title,
        stats_figT
    )
    
    # The following code correspodns to the Bokeh plots for the coefficients of the different angles in the model
    # The coefficients are arranged by tabs according to the assumption of the joint angle modeled
    # For instances only limbs have synergy assumption, so for spine joints this tab is empty
    
    # Users can drag the points on the plots to modify the coefficients, and then save the changes to make an effect on the prediction
    # Check the tools of the bokeh plot to interact with the plots
    
    # Assumption 1: Time-dependent transition
    dfCoef_A1 = dfCoef.filter(regex=selectAngle+'_'+angMod+'rotation')
    if trained_model != 'KF-GOM':
        df_A1 = pd.DataFrame(dfCoef_A1.columns, columns=['Variables'])
        dfCoef_A1_original = dfCoef_A1.copy()
        angleA1 = dfCoef_A1.columns.tolist()[0]
        
        n = len(dfCoef_A1.iloc[offset_time:,0].index) // 45
        x_markerA1 = dfCoef_A1.iloc[offset_time:,0].index[::n]
        y_markerA1 = dfCoef_A1.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values[::n]
        x_markerA1 = np.concatenate(([dfCoef_A1.iloc[offset_time:,0].index[0]], x_markerA1, [dfCoef_A1.iloc[offset_time:,0].index[-1]]))
        y_markerA1 = np.concatenate(([dfCoef_A1.iloc[offset_time:,0].values[0]], y_markerA1, [dfCoef_A1.iloc[offset_time:,0].values[-1]]))

        data_markerA1 = ColumnDataSource({
            'x': x_markerA1,
            'y': y_markerA1,
        })

        pA1 = figure(title=variables[0], x_axis_label='Frames', y_axis_label=dfCoef_A1.columns[0], plot_height=400)
        sourceA1 = ColumnDataSource(data=dict(x=dfCoef_A1.iloc[offset_time:,0].index, y=dfCoef_A1.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA1.line('x', 'y', source=sourceA1, line_width=2, color='black')

        rendererA1 = pA1.scatter(x='x', y='y', source=data_markerA1, color='darkgray', size=10, line_color='black')
        pA1.line(x='x', y='y', source=data_markerA1, color='darkgray')

        draw_toolA1 = PointDrawTool(renderers=[rendererA1], empty_value='black')
        pA1.add_tools(draw_toolA1)
        box_selectA1 = BoxSelectTool(renderers=[rendererA1])
        lasso_selectA1 = LassoSelectTool(renderers=[rendererA1])
        pA1.add_tools(box_selectA1, lasso_selectA1)
        plot_paneA1 = pn.pane.Bokeh(pA1)
        checkbox_groupA1 = pn.widgets.Tabulator(df_A1, selection=list(df_A1.index), height=400, selectable='checkbox')

        save_buttonA1 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_JbuttonA1 = pn.widgets.Button(name="Reset Angle Changes", button_type="warning", width=width_A_buttons)
        reset_AbuttonA1 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA1 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA1 = pn.Row(checkbox_groupA1, plot_paneA1, pn.Column(save_buttonA1, reset_JbuttonA1, reset_AbuttonA1, accept_buttonA1))
        data_marker_changesA1 = {}

        data_markerA1.on_change('data', update_data_marker_changesA1)
        checkbox_groupA1.on_click(plot_selected_angleA1)
        save_buttonA1.on_click(save_changesA1)
        reset_JbuttonA1.on_click(reset_jointA1)
        reset_AbuttonA1.on_click(reset_changesA1)
        accept_buttonA1.on_click(accept_changesA1)
        accept_changesA1(event=None)
    else: # For KF-GOM the constant coefficients are shown as sliders
        kf_variables_A1 = [KF_Variable(name=i, coef=dfCoef_A1[i], bounds=(0, dfCoef_A1[i]+10) if dfCoef_A1[i] >= 0 else (dfCoef_A1[i]-10, 0)) for i in list(dfCoef_A1.index)]
        dashboard_A1 = pn.Column(height=400, width=650)

        for i, var in enumerate(kf_variables_A1):
            dashboard_A1.append(
                pn.Row(
                    pn.pane.Markdown(f'**{var.name}**', width=200), 
                    var.param.value, 
                    var.reset_button,
                    align='center'
                )
            )

        save_buttonA1 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_AbuttonA1 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA1 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA1 = pn.Row(dashboard_A1, pn.Column(save_buttonA1, reset_AbuttonA1, accept_buttonA1), height=400, width=1000)

        save_buttonA1.on_click(save_changesA1_KF)
        reset_AbuttonA1.on_click(reset_changesA1_KF)
        accept_buttonA1.on_click(accept_changesA1_KF)
        accept_changesA1_KF(event=None)

    

    # Assumption 2: Intra-joint association
    dfCoef_A2 = dfCoef.filter(regex=selectAngle+'_')
    if trained_model != 'KF-GOM':
        columns_to_drop = dfCoef_A2.filter(regex=angMod).columns
        dfCoef_A2 = dfCoef_A2.drop(columns=columns_to_drop)
        df_A2 = pd.DataFrame(dfCoef_A2.columns, columns=['Variables'])
        dfCoef_A2_original = dfCoef_A2.copy()
        angleA2 = dfCoef_A2.columns.tolist()[0]
        
        n = len(dfCoef_A2.iloc[offset_time:,0].index) // 45
        x_markerA2 = dfCoef_A2.iloc[offset_time:,0].index[::n]
        y_markerA2 = dfCoef_A2.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values[::n]
        x_markerA2 = np.concatenate(([dfCoef_A2.iloc[offset_time:,0].index[0]], x_markerA2, [dfCoef_A2.iloc[offset_time:,0].index[-1]]))
        y_markerA2 = np.concatenate(([dfCoef_A2.iloc[offset_time:,0].values[0]], y_markerA2, [dfCoef_A2.iloc[offset_time:,0].values[-1]]))

        data_markerA2 = ColumnDataSource({
            'x': x_markerA2,
            'y': y_markerA2,
        })

        pA2 = figure(title=variables[0], x_axis_label='Frames', y_axis_label=dfCoef_A2.columns[0], plot_height=400)
        sourceA2 = ColumnDataSource(data=dict(x=dfCoef_A2.iloc[offset_time:,0].index, y=dfCoef_A2.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA2.line('x', 'y', source=sourceA2, line_width=2, color='salmon')

        rendererA2 = pA2.scatter(x='x', y='y', source=data_markerA2, color='darkred', size=10, line_color='black')
        pA2.line(x='x', y='y', source=data_markerA2, color='darkred')

        draw_toolA2 = PointDrawTool(renderers=[rendererA2], empty_value='black')
        pA2.add_tools(draw_toolA2)
        box_selectA2 = BoxSelectTool(renderers=[rendererA2])
        lasso_selectA2 = LassoSelectTool(renderers=[rendererA2])
        pA2.add_tools(box_selectA2, lasso_selectA2)
        plot_paneA2 = pn.pane.Bokeh(pA2)
        checkbox_groupA2 = pn.widgets.Tabulator(df_A2, selection=list(df_A2.index), height=400, selectable='checkbox')

        save_buttonA2 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_JbuttonA2 = pn.widgets.Button(name="Reset Angle Changes", button_type="warning", width=width_A_buttons)
        reset_AbuttonA2 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA2 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA2 = pn.Row(checkbox_groupA2, plot_paneA2, pn.Column(save_buttonA2, reset_JbuttonA2, reset_AbuttonA2, accept_buttonA2))
        data_marker_changesA2 = {}

        data_markerA2.on_change('data', update_data_marker_changesA2)
        checkbox_groupA2.on_click(plot_selected_angleA2)
        save_buttonA2.on_click(save_changesA2)
        reset_JbuttonA2.on_click(reset_jointA2)
        reset_AbuttonA2.on_click(reset_changesA2)
        accept_buttonA2.on_click(accept_changesA2)
        accept_changesA2(event=None)
    else: # For KF-GOM the constant coefficients are shown as sliders
        rows_to_drop = dfCoef_A2.filter(regex=angMod).index
        dfCoef_A2 = dfCoef_A2.drop(index=rows_to_drop)
        kf_variables_A2 = [KF_Variable(name=i, coef=dfCoef_A2[i], bounds=(0, dfCoef_A2[i]+10) if dfCoef_A2[i] >= 0 else (dfCoef_A2[i]-10, 0)) for i in list(dfCoef_A2.index)]   
        dashboard_A2 = pn.GridSpec(sizing_mode='stretch_both', max_height=300)
        dashboard_A2 = pn.Column(height=400, width=650)

        for i, var in enumerate(kf_variables_A2):
            dashboard_A2.append(
                pn.Row(
                    pn.pane.Markdown(f'**{var.name}**', width=200), 
                    var.param.value, 
                    var.reset_button,
                    align='center'
                )
            )


        save_buttonA2 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_AbuttonA2 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA2 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA2 = pn.Row(dashboard_A2, pn.Column(save_buttonA2, reset_AbuttonA2, accept_buttonA2), width=1000)

        save_buttonA2.on_click(save_changesA2_KF)
        reset_AbuttonA2.on_click(reset_changesA2_KF)
        accept_buttonA2.on_click(accept_changesA2_KF)
        accept_changesA2_KF(event=None)


    # Assumption 3: Inter-limb synergies
    if selectAngle not in nosynergy: # Only for joints with synergy assumption and arrange joint angles according if it is a right or left limb joint
        if selectAngle.startswith('Right'):
            selectAngleC = selectAngle.replace('Right', 'Left')
        elif selectAngle.startswith('Left'):
            selectAngleC = selectAngle.replace('Left', 'Right')

        dfCoef_A3 = dfCoef.filter(regex=selectAngleC)
        if trained_model != 'KF-GOM':
            df_A3 = pd.DataFrame(dfCoef_A3.columns, columns=['Variables'])
            dfCoef_A3_original = dfCoef_A3.copy()
            angleA3 = dfCoef_A3.columns.tolist()[0]
            
            n = len(dfCoef_A3.iloc[offset_time:,0].index) // 45
            x_markerA3 = dfCoef_A3.iloc[offset_time:,0].index[::n]
            y_markerA3 = dfCoef_A3.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values[::n]
            x_markerA3 = np.concatenate(([dfCoef_A3.iloc[offset_time:,0].index[0]], x_markerA3, [dfCoef_A3.iloc[offset_time:,0].index[-1]]))
            y_markerA3 = np.concatenate(([dfCoef_A3.iloc[offset_time:,0].values[0]], y_markerA3, [dfCoef_A3.iloc[offset_time:,0].values[-1]]))

            data_markerA3 = ColumnDataSource({
                'x': x_markerA3,
                'y': y_markerA3,
            })

            pA3 = figure(title=variables[0], x_axis_label='Frames', y_axis_label=dfCoef_A3.columns[0], plot_height=400)
            sourceA3 = ColumnDataSource(data=dict(x=dfCoef_A3.iloc[offset_time:,0].index, y=dfCoef_A3.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
            pA3.line('x', 'y', source=sourceA3, line_width=2, color='lightblue')

            rendererA3 = pA3.scatter(x='x', y='y', source=data_markerA3, color='midnightblue', size=10, line_color='black')
            pA3.line(x='x', y='y', source=data_markerA3, color='midnightblue')

            draw_toolA3 = PointDrawTool(renderers=[rendererA3], empty_value='black')
            pA3.add_tools(draw_toolA3)
            box_selectA3 = BoxSelectTool(renderers=[rendererA3])
            lasso_selectA3 = LassoSelectTool(renderers=[rendererA3])
            pA3.add_tools(box_selectA3, lasso_selectA3)
            plot_paneA3 = pn.pane.Bokeh(pA3)
            checkbox_groupA3 = pn.widgets.Tabulator(df_A3, selection=list(df_A3.index), height=400, selectable='checkbox')

            save_buttonA3 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
            reset_JbuttonA3 = pn.widgets.Button(name="Reset Angle Changes", button_type="warning", width=width_A_buttons)
            reset_AbuttonA3 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
            accept_buttonA3 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
            tabA3 = pn.Row(checkbox_groupA3, plot_paneA3, pn.Column(save_buttonA3, reset_JbuttonA3, reset_AbuttonA3, accept_buttonA3))
            data_marker_changesA3 = {}

            data_markerA3.on_change('data', update_data_marker_changesA3)
            checkbox_groupA3.on_click(plot_selected_angleA3)
            save_buttonA3.on_click(save_changesA3)
            reset_JbuttonA3.on_click(reset_jointA3)
            reset_AbuttonA3.on_click(reset_changesA3)
            accept_buttonA3.on_click(accept_changesA3)
            accept_changesA3(event=None)
        else: # For KF-GOM the constant coefficients are shown as sliders
            kf_variables_A3 = [KF_Variable(name=i, coef=dfCoef_A3[i], bounds=(0, dfCoef_A3[i]+10) if dfCoef_A3[i] >= 0 else (dfCoef_A3[i]-10, 0)) for i in list(dfCoef_A3.index)]
            dashboard_A3 = pn.Column(height=400, width=650)

            for i, var in enumerate(kf_variables_A3):
                dashboard_A3.append(
                    pn.Row(
                        pn.pane.Markdown(f'**{var.name}**', width=200), 
                        var.param.value, 
                        var.reset_button,
                        align='center'
                    )
                )
            save_buttonA3 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
            reset_JbuttonA3 = pn.widgets.Button(name="Reset Angle Changes", button_type="warning", width=width_A_buttons)
            reset_AbuttonA3 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
            accept_buttonA3 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
            tabA3 = pn.Row(dashboard_A3, pn.Column(save_buttonA3, reset_AbuttonA3, accept_buttonA3), width=1000)

            save_buttonA3.on_click(save_changesA3_KF)
            reset_AbuttonA3.on_click(reset_changesA3_KF)
            accept_buttonA3.on_click(accept_changesA3_KF)
            accept_changesA3_KF(event=None)
    else:
        tabA3 = pn.Row()
        dfCoef_A3 = pd.DataFrame()
        dfCoef_A3_original = dfCoef_A3.copy()
        coef_A3_GOM = pd.DataFrame()

    # Assumption 4.1: Serial Intra-limb mediation
    if positions[0].skeleton[selectAngle]['children'] == None:
            listA41 = positions[0].skeleton[selectAngle]['parent'] 
    elif positions[0].skeleton[selectAngle]['parent'] == None:
            listA41 = positions[0].skeleton[selectAngle]['children']
    else:
        listA41 = positions[0].skeleton[selectAngle]['children'] + [positions[0].skeleton[selectAngle]['parent']]
    
    if trained_model != 'KF-GOM':
        dfCoef_A41 = dfCoef[dfCoef.columns[dfCoef.columns.to_series().str.contains('|'.join(listA41))]]
        df_A41 = pd.DataFrame(dfCoef_A41.columns, columns=['Variables'])
        dfCoef_A41_original = dfCoef_A41.copy()
        angleA41 = dfCoef_A41.columns.tolist()[0]
        
        n = len(dfCoef_A41.iloc[offset_time:,0].index) // 45
        x_markerA41 = dfCoef_A41.iloc[offset_time:,0].index[::n]
        y_markerA41 = dfCoef_A41.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values[::n]
        x_markerA41 = np.concatenate(([dfCoef_A41.iloc[offset_time:,0].index[0]], x_markerA41, [dfCoef_A41.iloc[offset_time:,0].index[-1]]))
        y_markerA41 = np.concatenate(([dfCoef_A41.iloc[offset_time:,0].values[0]], y_markerA41, [dfCoef_A41.iloc[offset_time:,0].values[-1]]))

        data_markerA41 = ColumnDataSource({
            'x': x_markerA41,
            'y': y_markerA41,
        })

        pA41 = figure(title=variables[0], x_axis_label='Frames', y_axis_label=dfCoef_A41.columns[0], plot_height=400)
        sourceA41 = ColumnDataSource(data=dict(x=dfCoef_A41.iloc[offset_time:,0].index, y=dfCoef_A41.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA41.line('x', 'y', source=sourceA41, line_width=2, color='lightgreen')

        rendererA41 = pA41.scatter(x='x', y='y', source=data_markerA41, color='darkgreen', size=10, line_color='black')
        pA41.line(x='x', y='y', source=data_markerA41, color='darkgreen')

        draw_toolA41 = PointDrawTool(renderers=[rendererA41], empty_value='black')
        pA41.add_tools(draw_toolA41)
        box_selectA41 = BoxSelectTool(renderers=[rendererA41])
        lasso_selectA41 = LassoSelectTool(renderers=[rendererA41])
        pA41.add_tools(box_selectA41, lasso_selectA41)
        plot_paneA41 = pn.pane.Bokeh(pA41)
        checkbox_groupA41 = pn.widgets.Tabulator(df_A41, selection=list(df_A41.index), height=400, selectable='checkbox')

        save_buttonA41 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_JbuttonA41 = pn.widgets.Button(name="Reset Angle Changes", button_type="warning", width=width_A_buttons)
        reset_AbuttonA41 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA41 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA41 = pn.Row(checkbox_groupA41, plot_paneA41, pn.Column(save_buttonA41, reset_JbuttonA41, reset_AbuttonA41, accept_buttonA41))
        data_marker_changesA41 = {}

        data_markerA41.on_change('data', update_data_marker_changesA41)
        checkbox_groupA41.on_click(plot_selected_angleA41)
        save_buttonA41.on_click(save_changesA41)
        reset_JbuttonA41.on_click(reset_jointA41)
        reset_AbuttonA41.on_click(reset_changesA41)
        accept_buttonA41.on_click(accept_changesA41)
        accept_changesA41(event=None) 
    else: # For KF-GOM the constant coefficients are shown as sliders
        dfCoef_A41 = dfCoef[dfCoef.index[dfCoef.index.to_series().str.contains('|'.join(listA41))]]
        kf_variables_A41 = [KF_Variable(name=i, coef=dfCoef_A41[i], bounds=(0, dfCoef_A41[i]+10) if dfCoef_A41[i] >= 0 else (dfCoef_A41[i]-10, 0)) for i in list(dfCoef_A41.index)]
        dashboard_A41 = pn.Column(scroll = True, height=400, width=650)

        for i, var in enumerate(kf_variables_A41):
            dashboard_A41.append(
                pn.Row(
                    pn.pane.Markdown(f'**{var.name}**', width=200), 
                    var.param.value, 
                    var.reset_button,
                    align='center'
                )
            )
        
        save_buttonA41 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_AbuttonA41 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA41 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA41 = pn.Row(dashboard_A41, pn.Column(save_buttonA41, reset_AbuttonA41, accept_buttonA41), width=1000)

        save_buttonA41.on_click(save_changesA41_KF)
        reset_AbuttonA41.on_click(reset_changesA41_KF)
        accept_buttonA41.on_click(accept_changesA41_KF)
        accept_changesA41_KF(event=None)

    # Assumption 4.2: Non-Serial Intra-limb mediation
    if trained_model != 'KF-GOM':
        listA42 = dfCoef_A1.columns.tolist()+ dfCoef_A2.columns.tolist() + dfCoef_A3.columns.tolist()+ dfCoef_A41.columns.tolist()
        dfCoef_A42 = dfCoef.drop(columns=listA42)
        df_A42 = pd.DataFrame(dfCoef_A42.columns, columns=['Variables'])
        dfCoef_A42_original = dfCoef_A42.copy()
        angleA42 = dfCoef_A42.columns.tolist()[0]

        n = len(dfCoef_A42.iloc[offset_time:,0].index) // 45
        x_markerA42 = dfCoef_A42.iloc[offset_time:,0].index[::n]
        y_markerA42 = dfCoef_A42.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values[::n]
        x_markerA42 = np.concatenate(([dfCoef_A42.iloc[offset_time:,0].index[0]], x_markerA42, [dfCoef_A42.iloc[offset_time:,0].index[-1]]))
        y_markerA42 = np.concatenate(([dfCoef_A42.iloc[offset_time:,0].values[0]], y_markerA42, [dfCoef_A42.iloc[offset_time:,0].values[-1]]))

        data_markerA42 = ColumnDataSource({
            'x': x_markerA42,
            'y': y_markerA42,
        })

        pA42 = figure(title=variables[0], x_axis_label='Frames', y_axis_label=dfCoef_A42.columns[0], plot_height=400)
        sourceA42 = ColumnDataSource(data=dict(x=dfCoef_A42.iloc[offset_time:,0].index, y=dfCoef_A42.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA42.line('x', 'y', source=sourceA42, line_width=2, color='lightblue')

        rendererA42 = pA42.scatter(x='x', y='y', source=data_markerA42, color='midnightblue', size=10, line_color='black')
        pA42.line(x='x', y='y', source=data_markerA42, color='midnightblue')

        draw_toolA42 = PointDrawTool(renderers=[rendererA42], empty_value='black')
        pA42.add_tools(draw_toolA42)
        box_selectA42 = BoxSelectTool(renderers=[rendererA42])
        lasso_selectA42 = LassoSelectTool(renderers=[rendererA42])
        pA42.add_tools(box_selectA42, lasso_selectA42)
        plot_paneA42 = pn.pane.Bokeh(pA42)
        checkbox_groupA42 = pn.widgets.Tabulator(df_A42, selection=list(df_A42.index), height=400, selectable='checkbox')

        save_buttonA42 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_JbuttonA42 = pn.widgets.Button(name="Reset Angle Changes", button_type="warning", width=width_A_buttons)
        reset_AbuttonA42 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA42 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA42 = pn.Row(checkbox_groupA42, plot_paneA42, pn.Column(save_buttonA42, reset_JbuttonA42, reset_AbuttonA42, accept_buttonA42))
        data_marker_changesA42 = {}

        data_markerA42.on_change('data', update_data_marker_changesA42)
        checkbox_groupA42.on_click(plot_selected_angleA42)
        save_buttonA42.on_click(save_changesA42)
        reset_JbuttonA42.on_click(reset_jointA42)
        reset_AbuttonA42.on_click(reset_changesA42)
        accept_buttonA42.on_click(accept_changesA42)
        accept_changesA42(event=None) 
    else: # For KF-GOM the constant coefficients are shown as sliders
        listA42 = dfCoef_A1.index.tolist()+ dfCoef_A2.index.tolist() + dfCoef_A3.index.tolist()+ dfCoef_A41.index.tolist()
        dfCoef_A42 = dfCoef.drop(index=listA42)
        kf_variables_A42 = [KF_Variable(name=i, coef=dfCoef_A42[i], bounds=(0, dfCoef_A42[i]+10) if dfCoef_A42[i] >= 0 else (dfCoef_A42[i]-10, 0)) for i in list(dfCoef_A42.index)]
        dashboard_A42 = pn.Column(scroll = True, height=400, width=650)

        for i, var in enumerate(kf_variables_A42):
            if var.name == 'Bias':
                continue
            else:
                dashboard_A42.append(
                    pn.Row(
                        pn.pane.Markdown(f'**{var.name}**', width=200), 
                        var.param.value, 
                        var.reset_button,
                        align='center'
                    )
                )
        save_buttonA42 = pn.widgets.Button(name="Save Changes", button_type='primary', width=width_A_buttons)
        reset_AbuttonA42 = pn.widgets.Button(name="Reset All Changes", button_type="warning", width=width_A_buttons)
        accept_buttonA42 = pn.widgets.Button(name="Accept Changes", button_type='primary', width=width_A_buttons)
        tabA42 = pn.Row(dashboard_A42, pn.Column(save_buttonA42, reset_AbuttonA42, accept_buttonA42), width=1000)

        save_buttonA42.on_click(save_changesA42_KF)
        reset_AbuttonA42.on_click(reset_changesA42_KF)
        accept_buttonA42.on_click(accept_changesA42_KF)
        accept_changesA42_KF(event=None)

    tabsC = pn.Tabs(('Transitioning', tabA1), ('Intra-joint association', tabA2), ('Inter-limb synergy', tabA3), 
                    ('Serial intra-limb mediation', tabA41),('Non-serial intra-limb mediation', tabA42), 
                    ('All assumptions statistics', col_stats), height=400)

    # Visualization of the main tab in the GOM card
    columnCard = pn.Column('## Training',
                           pn.layout.Divider(margin=(-20, 0, 0, 0)),
                           toggle_Model_Widgets, 
                           '## Prediction',
                           pn.layout.Divider(margin=(-20, 0, 0, 0)),
                           selectAngleCoef, 
                           '## Assumptions',
                           pn.layout.Divider(margin=(-20, 0, 0, 0)),
                           tabsC, 
                           width=1100, height=850)
    

    #%% Generated movement tab in the GOM card

    fig_pred = go.Figure(
        layout=go.Layout(
            autosize=False,
            width=600,
            height=400
        )
    )
    fig_pred.add_trace(go.Scatter(
            x=np.arange(0, len(pred.iloc[offset_time+3:,0])), 
            y=pred.loc[offset_time+3:, selectAngle+'_'+angMod+'rotation'].rolling(20, min_periods=1).mean(), 
            mode='lines', name='Prediction', line=dict(color='red')))
    fig_pred.add_trace(go.Scatter(
            x=np.arange(0, len(eulerAngles.iloc[:len(pred.iloc[offset_time+3:,0]),0])), 
            y=eulerAngles.loc[:len(pred.iloc[offset_time+3:,0]), selectAngle+'_'+angMod+'rotation'], 
            mode='lines', name='Original', line=dict(dash='dash', color='darkgray')))
    fig_pred.update_xaxes(title_text='Time frame')
    fig_pred.update_yaxes(title_text='Angle value')
    fig_pred.update_layout(title={'text': selectAngle+'_'+angMod+'rotation', 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
    fig_pred_plot = pn.pane.Plotly(fig_pred, align = 'center')


    ## Plot predicted skeleton with the trajectory of the modeled joint angle
    parsed_dataPred = parsed_data
    dfVal = parsed_dataPred.values
    nn = [dfVal.columns.get_loc(c) for c in pred.columns if c in dfVal]
    first_val = dfVal.iloc[0:3,nn]
    df_val = pd.concat([first_val, pred]).reset_index(drop = True)
    parsed_dataPred.values.iloc[:,nn] = df_val
    positionsPred = mp.fit_transform([parsed_dataPred])
    fig_pred_skel = draw_stickfigure3d_js(positionsPred[0], frame=0, highlight_joint = selectAngle, GOM_Skel=GOM_Skel)

    trajX = positionsPred[0].values.loc[:,selectAngle+'_Xposition'] - positionsPred[0].values.loc[:,'Hips_Xposition']
    trajY = positionsPred[0].values.loc[:,selectAngle+'_Zposition'] - positionsPred[0].values.loc[:,'Hips_Zposition']
    trajZ = positionsPred[0].values.loc[:,selectAngle+'_Yposition'] - positionsPred[0].values.loc[:,'Hips_Yposition'] + positionsPred[0].values['RightFootToe_Yposition'][0]
    
    fig_pred_skel.add_trace(go.Scatter3d(x=trajX.iloc[offset_time:], y=trajY.iloc[offset_time:],z=trajZ.iloc[offset_time:], mode='lines', 
                                 name='Predicted Trajectory', line=dict(color='red')))
    
    trajX = positions[0].values.loc[:,selectAngle+'_Xposition'] - positions[0].values.loc[:,'Hips_Xposition']
    trajY = positions[0].values.loc[:,selectAngle+'_Zposition'] - positions[0].values.loc[:,'Hips_Zposition']
    trajZ = positions[0].values.loc[:,selectAngle+'_Yposition'] - positions[0].values.loc[:,'Hips_Yposition'] + positions[0].values['RightFootToe_Yposition'][0]
    
    
    fig_pred_skel.add_trace(go.Scatter3d(x=trajX.iloc[offset_time:], y=trajY.iloc[offset_time:],z=trajZ.iloc[offset_time:], mode='lines', 
                                 name='Original Trajectory', line=dict(color='darkgray')))

    fig_pred_skel.update_layout(autosize=False, width=400, height=400)

    figA_pred_pane = pn.pane.Plotly(fig_pred_skel)

    # This predicted skeleton has it own slider to change the frame
    frame_slider_pred = pn.widgets.EditableIntSlider(name='Frame', start=1, end=len(pred)-offset_time, step=1, value=0)
    frame_slider_pred.param.watch(update_frame_pred, 'value')

    # Visualization of the generated movement tab in the GOM card
    pred_skel_tools = pn.Column(figA_pred_pane, frame_slider_pred)
    fig_pred_pane = pn.Row(fig_pred_plot, pred_skel_tools)

    
    # Run all initial visualization of the GOM card
    accept_changes(event=None)

    if trained_model != 'KF-GOM':
        coef_i = pd.concat([coef_A1_GOM, coef_A2_GOM, coef_A3_GOM, coef_A41_GOM, coef_A42_GOM], axis=1)
        coef_init = coef_i[col_coef].reset_index(drop=True)
        init_col = coef_init.columns.tolist()
        coef_init_mod = coef.copy()
        coef_init_mod[init_col] = coef_init[init_col]
    else:
        if coef_A3_GOM.empty:
            coef_i = pd.concat([coef_A1_GOM, coef_A2_GOM, coef_A41_GOM, coef_A42_GOM], axis=0)
        else:
            coef_i = pd.concat([coef_A1_GOM, coef_A2_GOM, coef_A3_GOM, coef_A41_GOM, coef_A42_GOM], axis=0)
        coef_i = coef_i[row_coef]
        coef_init_mod = coef.copy()
        coef_init_mod.iloc[:,index] = coef_i

    if toggle_Model.value == 'ATT-RGOM':
        predOrg =  ae_gom.pred_ang_coef(eulerAngles_original, coef)
        pred_i =  ae_gom.pred_ang_coef(eulerAngles_original, coef_init_mod)
    elif toggle_Model.value == 'VAE-RGOM':
        predOrg =  vae_gom.pred_ang_coef(eulerAngles_original, coef)
        pred_i =  vae_gom.pred_ang_coef(eulerAngles_original, coef_init_mod)
    elif toggle_Model.value == 'T-RGOM': 
        predOrg =  t_gom.pred_ang_coef(eulerAngles_original, coef)
        pred_i =  t_gom.pred_ang_coef(eulerAngles_original, coef_init_mod)
    elif toggle_Model.value == 'KF-GOM': 
        predOrg =  kf_gom.pred_ang_coef(eulerAngles_original, coef)
        pred_i =  kf_gom.pred_ang_coef(eulerAngles_original, coef_init_mod)

    offsetP = pred_i - predOrg


    plot_selected_angle(event=None)
    
    # Visualization of the GOM card:
    tabJA = pn.Row(checkbox_group, plot_pane, pn.Column(save_button, reset_Jbutton, reset_Abutton, accept_button), width=1100, height=600)

    tabsGOM = pn.Tabs(('Joint Angles', tabJA), ('Model', columnCard),('Generated Movement', fig_pred_pane), active=1)

    # Whenever the user do changes in the Model or Joint Angles tab, he/she needs to press the Predict Joint Angle button 
    # to see the changes in the Generated Movement tab
    predict_button = pn.widgets.Button(name="Predict Joint Angle", button_type='primary', width=width_A_buttons, align='center')
    predict_button.on_click(predict_angle)

    # I added an extra button to download the generated data as a csv file
    sio = StringIO()
    pred_mod = pred_i
    pred_mod.to_csv(sio)
    sio.seek(0)
    download_Gdat_button = pn.widgets.FileDownload(sio, embed=True, label="Download Generated Data", width=200, button_style='solid', filename='GOM_prediction.csv')

    buttons_pred_dw = pn.Row(predict_button, download_Gdat_button)

    # The final card of the GOM
    cardGOM = pn.Card(tabsGOM, buttons_pred_dw, title='Gesture Operational Model')

    # If the user wants to compare coefficients, joint angles and precitions with a second recording, the following code is executed
    if compare == True:
        plot_iGOM_comp()
    
    return cardGOM

# Function to plot the trajectories and skeleton of the generated movement tab from the GOM card
def update_frame_pred(event):
    camA = figA_pred_pane.object.layout['scene']['camera']
    fig_pred_skel = draw_stickfigure3d_js(positionsPred[0], frame=event.new, cam = camA, highlight_joint = selectAngle.value, GOM_Skel=GOM_Skel)

    trajX = positionsPred[0].values.loc[:,selectAngle.value+'_Xposition'] - positionsPred[0].values.loc[:,'Hips_Xposition']
    trajY = positionsPred[0].values.loc[:,selectAngle.value+'_Zposition'] - positionsPred[0].values.loc[:,'Hips_Zposition']
    trajZ = positionsPred[0].values.loc[:,selectAngle.value+'_Yposition'] - positionsPred[0].values.loc[:,'Hips_Yposition'] + positionsPred[0].values['RightFootToe_Yposition'][0]
    
    fig_pred_skel.add_trace(go.Scatter3d(x=trajX.iloc[offset_time:], y=trajY.iloc[offset_time:],z=trajZ.iloc[offset_time:], mode='lines', 
                                 name='Predicted Trajectory'))
    
    trajX = positions[0].values.loc[:,selectAngle.value+'_Xposition'] - positions[0].values.loc[:,'Hips_Xposition']
    trajY = positions[0].values.loc[:,selectAngle.value+'_Zposition'] - positions[0].values.loc[:,'Hips_Zposition']
    trajZ = positions[0].values.loc[:,selectAngle.value+'_Yposition'] - positions[0].values.loc[:,'Hips_Yposition'] + positions[0].values['RightFootToe_Yposition'][0]
    
    
    fig_pred_skel.add_trace(go.Scatter3d(x=trajX.iloc[offset_time:], y=trajY.iloc[offset_time:],z=trajZ.iloc[offset_time:], mode='lines', 
                                 name='Original Trajectory', line=dict(color='darkgray')))

    fig_pred_skel.update_layout(autosize=False, width=400, height=400)
    figA_pred_pane.object = fig_pred_skel
    figA_pred_pane.param.trigger('object')

# Plot of the angle in the Joint Angles tab from the GOM card
def plot_selected_angle(event):
    global jointA
    if event is None:
        jointA = variables[0]
    else:
        jointA = variables[event.row]
    selected_angleOrg = eulerAngles_original.loc[offset_time:,jointA]
    selected_angle = eulerAngles.loc[offset_time:,jointA]

    n = len(selected_angle.index) // 45
    x_marker = selected_angle.index[::n]
    y_marker = selected_angle.values[::n]

    # Add the first and last points of the sine curve to the markers
    x_marker = np.concatenate(([selected_angle.index[0]], x_marker, [selected_angle.index[-1]]))
    y_marker = np.concatenate(([selected_angle.values[0]], y_marker, [selected_angle.values[-1]]))

    p.title.text = jointA
    source.data = dict(x=selected_angleOrg.index, y=selected_angleOrg.values)
    data_marker.data = dict(x=x_marker, y=y_marker)
    if compare == True:
        source_comp.data = dict(x=eulerAngles_comp.loc[offset_time:,jointA].index, y=eulerAngles_comp.loc[offset_time:,jointA].values)


############################################################################################################
# The following functions are used to update the data marker changes in the Joint Angle and Model tabs from the GOM card 
# First are the functions forchanges in the Joint Angle tab and 
# then the functions for changes in the Model tab (time-varying coefficients from A1, A2, A3, A41, A42)

def update_data_marker_changes(attr, old, new):
    data_marker_changes[jointA] = {'x': new['x'], 'y': new['y']}

def save_changes(event):
    for joint_angle, changes in data_marker_changes.items():
        # Interpolate the changes to match the index of eulerAngles
        new_values = np.interp(eulerAngles.iloc[offset_time:,:].index, changes['x'], changes['y'])
        eulerAngles.loc[offset_time:,joint_angle] = pd.Series(index=eulerAngles.iloc[offset_time:,:].index, data=new_values)
    #print("Changes saved.")

def reset_changes(event):
    for joint_angle in data_marker_changes.keys():
        eulerAngles.loc[:,joint_angle] = eulerAngles_original[joint_angle]
    #print("Changes reset.")
    
    # Clear the data_marker_changes dictionary
    data_marker_changes.clear()
    # Reset the data_marker data source
    initial_angle = eulerAngles[variables[0]].iloc[offset_time:]
    # Select every 5th point for the markers
    n = len(initial_angle.index) // 45
    x_marker = initial_angle.index[::n]
    y_marker = initial_angle.values[::n]

    # Add the first and last points of the sine curve to the markers
    x_marker = np.concatenate(([initial_angle.index[0]], x_marker, [initial_angle.index[-1]]))
    y_marker = np.concatenate(([initial_angle.values[0]], y_marker, [initial_angle.values[-1]]))
    
    p.title.text = variables[0]
    selected_angleOrg = eulerAngles_original[variables[0]].iloc[offset_time:]
    source.data = dict(x=selected_angleOrg.index, y=selected_angleOrg.values)
    data_marker.data = dict(x=x_marker, y=y_marker)

def reset_joint(event):
    eulerAngles.loc[:,jointA] = eulerAngles_original[jointA]
    #print(f"Changes to {jointA} reset.")
    # Clear the data_marker_changes dictionary for the selected joint angle
    data_marker_changes.pop(jointA, None)
    # Reset the data_marker data source for the selected joint angle
    initial_angle = eulerAngles[jointA].iloc[offset_time:]
    # Select every 5th point for the markers
    n = len(initial_angle.index) // 45
    x_marker = initial_angle.index[::n]
    y_marker = initial_angle.values[::n]

    # Add the first and last points of the sine curve to the markers
    x_marker = np.concatenate(([initial_angle.index[0]], x_marker, [initial_angle.index[-1]]))
    y_marker = np.concatenate(([initial_angle.values[0]], y_marker, [initial_angle.values[-1]]))

    data_marker.data = dict(x=x_marker, y=y_marker)

def accept_changes(event):
    global eulerAngles_GOM
    # Get the selected joint angle
    eulerAngles_GOM = eulerAngles.copy()

    checked_angles = checkbox_group.selection
    checked_angles2 = [variables[row] for row in checked_angles]

    for joint_angle in variables:
        if joint_angle not in checked_angles2:
            eulerAngles_GOM[joint_angle] = np.zeros(len(eulerAngles_GOM))

## A1

def plot_selected_angleA1(event):
    global angleA1, dfCoef_A1_original, dfCoef_A1, pA1, sourceA1, data_markerA1
    angleA1 = dfCoef_A1.columns.tolist()[event.row]
    selected_coefOrg = dfCoef_A1_original.loc[offset_time:,angleA1].rolling(20, min_periods=1).mean()
    selected_coef = dfCoef_A1.loc[offset_time:,angleA1]

    n = len(dfCoef_A1.index) // 45
    x_markerA1 = selected_coef.index[::n]
    y_markerA1 = selected_coef.rolling(20, min_periods=1).mean().values[::n]
    x_markerA1 = np.concatenate(([selected_coef.index[0]], x_markerA1, [selected_coef.index[-1]]))
    y_markerA1 = np.concatenate(([selected_coef.values[0]], y_markerA1, [selected_coef.values[-1]]))

    pA1.title.text = angleA1
    data_markerA1.data = dict(x=x_markerA1, y=y_markerA1)
    sourceA1.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)

    if compare == True:
        sourceA1_comp.data = dict(x=dfCoef_A1_comp.loc[offset_time:,angleA1].index, y=dfCoef_A1_comp.loc[offset_time:,angleA1].values)

def update_data_marker_changesA1(attr, old, new):
    data_marker_changesA1[angleA1] = {'x': new['x'], 'y': new['y']}

def save_changesA1(event):
    for joint_angle, changes in data_marker_changesA1.items():
        new_values = np.interp(dfCoef_A1.iloc[offset_time:,:].index, changes['x'], changes['y'])
        dfCoef_A1.loc[offset_time:,joint_angle] = pd.Series(index=dfCoef_A1.iloc[offset_time:,:].index, data=new_values)
    #print("Changes A1 saved.")

def save_changesA1_KF(event):
    slider_values = [var.value for var in kf_variables_A1]
    # Update dfCoef_A1 according to the slider values
    dfCoef_A1.iloc[:] = slider_values
    #print("Changes A1 saved.")

def reset_changesA1(event):
    for joint_angle in data_marker_changesA1.keys():
        dfCoef_A1.loc[offset_time:,joint_angle] = dfCoef_A1_original.loc[offset_time:,joint_angle]
    #print("Changes A1 reset.")
    
    data_marker_changesA1.clear()
    initial_coef = dfCoef_A1[dfCoef_A1.columns.tolist()[0]].iloc[offset_time:]

    n = len(initial_coef.index) // 45
    x_markerA1 = initial_coef.index[::n]
    y_markerA1 = initial_coef.values[::n]
    x_markerA1 = np.concatenate(([initial_coef.index[0]], x_markerA1, [initial_coef.index[-1]]))
    y_markerA1 = np.concatenate(([initial_coef.values[0]], y_markerA1, [initial_coef.values[-1]]))
    
    pA1.title.text = dfCoef_A1.columns.tolist()[0]
    selected_coefOrg = dfCoef_A1_original[dfCoef_A1.columns.tolist()[0]].iloc[offset_time:].rolling(20, min_periods=1).mean()
    sourceA1.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)
    data_markerA1.data = dict(x=x_markerA1, y=y_markerA1)

def reset_changesA1_KF(event):
    for var in kf_variables_A1:
        var._reset()
    #print("Changes A1 reset.")

def reset_jointA1(event):
    dfCoef_A1.loc[offset_time:,angleA1] = dfCoef_A1_original[angleA1].iloc[offset_time:]
    #print(f"Changes to {angleA1} reset.")
    data_marker_changesA1.pop(angleA1, None)
    initial_angle = dfCoef_A1[angleA1].iloc[offset_time:]
    n = len(initial_angle.index) // 45
    x_markerA1 = initial_angle.index[::n]
    y_markerA1 = initial_angle.values[::n]

    x_markerA1 = np.concatenate(([initial_angle.index[0]], x_markerA1, [initial_angle.index[-1]]))
    y_markerA1 = np.concatenate(([initial_angle.values[0]], y_markerA1, [initial_angle.values[-1]]))
    data_markerA1.data = dict(x=x_markerA1, y=y_markerA1)

def accept_changesA1(event):
    global coef_A1_GOM
    coef_A1_GOM = dfCoef_A1.copy()
    checked_coefA1 = checkbox_groupA1.selection
    checked_coefA1_2 = [dfCoef_A1.columns.tolist()[row] for row in checked_coefA1]

    for joint_angle in dfCoef_A1.columns.tolist():
        if joint_angle not in checked_coefA1_2:
            coef_A1_GOM[joint_angle] = np.zeros(len(coef_A1_GOM))

def accept_changesA1_KF(event):
    global coef_A1_GOM
    coef_A1_GOM = dfCoef_A1.copy()

## A2

def plot_selected_angleA2(event):
    global angleA2, dfCoef_A2_original, dfCoef_A2, pA2, sourceA2, data_markerA2
    angleA2 = dfCoef_A2.columns.tolist()[event.row]
    selected_coefOrg = dfCoef_A2_original.loc[offset_time:,angleA2].rolling(20, min_periods=1).mean()
    selected_coef = dfCoef_A2.loc[offset_time:,angleA2]

    n = len(selected_coef.index) // 45
    x_markerA2 = selected_coef.index[::n]
    y_markerA2 = selected_coef.rolling(20, min_periods=1).mean().values[::n]
    x_markerA2 = np.concatenate(([selected_coef.index[0]], x_markerA2, [selected_coef.index[-1]]))
    y_markerA2 = np.concatenate(([selected_coef.values[0]], y_markerA2, [selected_coef.values[-1]]))

    pA2.title.text = angleA2
    data_markerA2.data = dict(x=x_markerA2, y=y_markerA2)
    sourceA2.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)

    if compare == True:
        sourceA2_comp.data = dict(x=dfCoef_A2_comp.loc[offset_time:,angleA2].index, y=dfCoef_A2_comp.loc[offset_time:,angleA2].values)

def update_data_marker_changesA2(attr, old, new):
    data_marker_changesA2[angleA2] = {'x': new['x'], 'y': new['y']}

def save_changesA2(event):
    for joint_angle, changes in data_marker_changesA2.items():
        new_values = np.interp(dfCoef_A2.iloc[offset_time:,:].index, changes['x'], changes['y'])
        dfCoef_A2.loc[offset_time:,joint_angle] = pd.Series(index=dfCoef_A2.iloc[offset_time:,:].index, data=new_values)
    #print("Changes A2 saved.")

def save_changesA2_KF(event):
    slider_values = [var.value for var in kf_variables_A2]
    # Update dfCoef_A1 according to the slider values
    dfCoef_A2[:] = slider_values
    #print("Changes A2 saved.")

def reset_changesA2(event):
    for joint_angle in data_marker_changesA2.keys():
        dfCoef_A2.loc[offset_time:,joint_angle] = dfCoef_A2_original.loc[offset_time:,joint_angle]
    #print("Changes A2 reset.")
    
    data_marker_changesA2.clear()
    initial_coef = dfCoef_A2[dfCoef_A2.columns.tolist()[0]].iloc[offset_time:]
    n = len(initial_coef.index) // 45
    x_markerA2 = initial_coef.index[::n]
    y_markerA2 = initial_coef.values[::n]
    x_markerA2 = np.concatenate(([initial_coef.index[0]], x_markerA2, [initial_coef.index[-1]]))
    y_markerA2 = np.concatenate(([initial_coef.values[0]], y_markerA2, [initial_coef.values[-1]]))
    
    pA2.title.text = dfCoef_A2.columns.tolist()[0]
    selected_coefOrg = dfCoef_A2_original[dfCoef_A2.columns.tolist()[0]].iloc[offset_time:].rolling(20, min_periods=1).mean()
    sourceA2.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)
    data_markerA2.data = dict(x=x_markerA2, y=y_markerA2)

def reset_changesA2_KF(event):
    for var in kf_variables_A2:
        var._reset()
    #print("Changes A2 reset.")

def reset_jointA2(event):
    dfCoef_A2.loc[offset_time:,angleA2] = dfCoef_A2_original[angleA2].iloc[offset_time:]
    #print(f"Changes to {angleA2} reset.")
    data_marker_changesA2.pop(angleA2, None)
    initial_angle = dfCoef_A2[angleA2].iloc[offset_time:]
    n = len(initial_angle.index) // 45
    x_markerA2 = initial_angle.index[::n]
    y_markerA2 = initial_angle.values[::n]

    x_markerA2 = np.concatenate(([initial_angle.index[0]], x_markerA2, [initial_angle.index[-1]]))
    y_markerA2 = np.concatenate(([initial_angle.values[0]], y_markerA2, [initial_angle.values[-1]]))
    data_markerA2.data = dict(x=x_markerA2, y=y_markerA2)

def accept_changesA2(event):
    global coef_A2_GOM
    coef_A2_GOM = dfCoef_A2.copy()
    checked_coefA2 = checkbox_groupA2.selection
    checked_coefA2_2 = [dfCoef_A2.columns.tolist()[row] for row in checked_coefA2]

    for joint_angle in dfCoef_A2.columns.tolist():
        if joint_angle not in checked_coefA2_2:
            coef_A2_GOM[joint_angle] = np.zeros(len(coef_A2_GOM))

def accept_changesA2_KF(event):
    global coef_A2_GOM
    coef_A2_GOM = dfCoef_A2.copy()

## A3

def plot_selected_angleA3(event):
    global angleA3, dfCoef_A3_original, dfCoef_A3, pA3, sourceA3, data_markerA3
    angleA3 = dfCoef_A3.columns.tolist()[event.row]
    selected_coefOrg = dfCoef_A3_original.loc[offset_time:,angleA3].rolling(20, min_periods=1).mean()
    selected_coef = dfCoef_A3.loc[offset_time:,angleA3]

    n = len(selected_coef.index) // 45

    x_markerA3 = selected_coef.index[::n]
    y_markerA3 = selected_coef.rolling(20, min_periods=1).mean().values[::n]
    x_markerA3 = np.concatenate(([selected_coef.index[0]], x_markerA3, [selected_coef.index[-1]]))
    y_markerA3 = np.concatenate(([selected_coef.values[0]], y_markerA3, [selected_coef.values[-1]]))

    pA3.title.text = angleA3
    data_markerA3.data = dict(x=x_markerA3, y=y_markerA3)
    sourceA3.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)

    if compare == True:
        sourceA3_comp.data = dict(x=dfCoef_A3_comp.loc[offset_time:,angleA3].index, y=dfCoef_A3_comp.loc[offset_time:,angleA3].values)

def update_data_marker_changesA3(attr, old, new):
    data_marker_changesA3[angleA3] = {'x': new['x'], 'y': new['y']}

def save_changesA3(event):
    for joint_angle, changes in data_marker_changesA3.items():
        new_values = np.interp(dfCoef_A3.iloc[offset_time:,:].index, changes['x'], changes['y'])
        dfCoef_A3.loc[offset_time:,joint_angle] = pd.Series(index=dfCoef_A3.iloc[offset_time:,:].index, data=new_values)
    #print("Changes A3 saved.")

def save_changesA3_KF(event):
    slider_values = [var.value for var in kf_variables_A3]
    # Update dfCoef_A1 according to the slider values
    dfCoef_A3[:] = slider_values
    #print("Changes A3 saved.")

def reset_changesA3(event):
    for joint_angle in data_marker_changesA3.keys():
        dfCoef_A3.loc[offset_time:,joint_angle] = dfCoef_A3_original.loc[offset_time:,joint_angle]
    #print("Changes A3 reset.")
    
    data_marker_changesA3.clear()
    initial_coef = dfCoef_A3[dfCoef_A3.columns.tolist()[0]].iloc[offset_time:]
    n = len(initial_coef.index) // 45
    x_markerA3 = initial_coef.index[::n]
    y_markerA3 = initial_coef.values[::n]
    x_markerA3 = np.concatenate(([initial_coef.index[0]], x_markerA3, [initial_coef.index[-1]]))
    y_markerA3 = np.concatenate(([initial_coef.values[0]], y_markerA3, [initial_coef.values[-1]]))
    
    pA3.title.text = dfCoef_A3.columns.tolist()[0]
    selected_coefOrg = dfCoef_A3_original[dfCoef_A3.columns.tolist()[0]].iloc[offset_time:].rolling(20, min_periods=1).mean()
    sourceA3.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)
    data_markerA3.data = dict(x=x_markerA3, y=y_markerA3)

def reset_changesA3_KF(event):
    for var in kf_variables_A3:
        var._reset()
    #print("Changes A3 reset.")

def reset_jointA3(event):
    dfCoef_A3.loc[offset_time:,angleA3] = dfCoef_A3_original[angleA3].iloc[offset_time:]
    #print(f"Changes to {angleA3} reset.")
    data_marker_changesA3.pop(angleA3, None)
    initial_angle = dfCoef_A3[angleA3].iloc[offset_time:]
    n = len(initial_angle.index) // 45
    x_markerA3 = initial_angle.index[::n]
    y_markerA3 = initial_angle.values[::n]

    x_markerA3 = np.concatenate(([initial_angle.index[0]], x_markerA3, [initial_angle.index[-1]]))
    y_markerA3 = np.concatenate(([initial_angle.values[0]], y_markerA3, [initial_angle.values[-1]]))
    data_markerA3.data = dict(x=x_markerA3, y=y_markerA3)

def accept_changesA3(event):
    global coef_A3_GOM
    coef_A3_GOM = dfCoef_A3.copy()
    checked_coefA3 = checkbox_groupA3.selection
    checked_coefA3_2 = [dfCoef_A3.columns.tolist()[row] for row in checked_coefA3]

    for joint_angle in dfCoef_A3.columns.tolist():
        if joint_angle not in checked_coefA3_2:
            coef_A3_GOM[joint_angle] = np.zeros(len(coef_A3_GOM))

def accept_changesA3_KF(event):
    global coef_A3_GOM
    coef_A3_GOM = dfCoef_A3.copy()

## A41

def plot_selected_angleA41(event):
    global angleA41, dfCoef_A41_original, dfCoef_A41, pA41, sourceA41, data_markerA41
    angleA41 = dfCoef_A41.columns.tolist()[event.row]
    selected_coefOrg = dfCoef_A41_original.loc[offset_time:,angleA41].rolling(20, min_periods=1).mean()
    selected_coef = dfCoef_A41.loc[offset_time:,angleA41]

    n = len(selected_coef.index) // 45

    x_markerA41 = selected_coef.index[::n]
    y_markerA41 = selected_coef.rolling(20, min_periods=1).mean().values[::n]
    x_markerA41 = np.concatenate(([selected_coef.index[0]], x_markerA41, [selected_coef.index[-1]]))
    y_markerA41 = np.concatenate(([selected_coef.values[0]], y_markerA41, [selected_coef.values[-1]]))

    pA41.title.text = angleA41
    data_markerA41.data = dict(x=x_markerA41, y=y_markerA41)
    sourceA41.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)

    if compare == True:
        sourceA41_comp.data = dict(x=dfCoef_A41_comp.loc[offset_time:,angleA41].index, y=dfCoef_A41_comp.loc[offset_time:,angleA41].values)

def update_data_marker_changesA41(attr, old, new):
    data_marker_changesA41[angleA41] = {'x': new['x'], 'y': new['y']}

def save_changesA41(event):
    for joint_angle, changes in data_marker_changesA41.items():
        new_values = np.interp(dfCoef_A41.iloc[offset_time:,:].index, changes['x'], changes['y'])
        dfCoef_A41.loc[offset_time:,joint_angle] = pd.Series(index=dfCoef_A41.iloc[offset_time:,:].index, data=new_values)
    #print("Changes A41 saved.")

def save_changesA41_KF(event):
    slider_values = [var.value for var in kf_variables_A41]
    # Update dfCoef_A1 according to the slider values
    dfCoef_A41[:] = slider_values
    #print("Changes A41 saved.")

def reset_changesA41(event):
    for joint_angle in data_marker_changesA41.keys():
        dfCoef_A41.loc[offset_time:,joint_angle] = dfCoef_A41_original.loc[offset_time:,joint_angle]
    #print("Changes A41 reset.")
    
    data_marker_changesA41.clear()
    initial_coef = dfCoef_A41[dfCoef_A41.columns.tolist()[0]].iloc[offset_time:]
    n = len(initial_coef.index) // 45
    x_markerA41 = initial_coef.index[::n]
    y_markerA41 = initial_coef.values[::n]
    x_markerA41 = np.concatenate(([initial_coef.index[0]], x_markerA41, [initial_coef.index[-1]]))
    y_markerA41 = np.concatenate(([initial_coef.values[0]], y_markerA41, [initial_coef.values[-1]]))
    
    pA41.title.text = dfCoef_A41.columns.tolist()[0]
    selected_coefOrg = dfCoef_A41_original[dfCoef_A41.columns.tolist()[0]].iloc[offset_time:].rolling(20, min_periods=1).mean()
    sourceA41.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)
    data_markerA41.data = dict(x=x_markerA41, y=y_markerA41)

def reset_changesA41_KF(event):
    for var in kf_variables_A41:
        var._reset()
    #print("Changes A41 reset.")

def reset_jointA41(event):
    dfCoef_A41.loc[offset_time:,angleA41] = dfCoef_A41_original[angleA41].iloc[offset_time:]
    #print(f"Changes to {angleA41} reset.")
    data_marker_changesA41.pop(angleA41, None)
    initial_angle = dfCoef_A41[angleA41].iloc[offset_time:]
    n = len(initial_angle.index) // 45
    x_markerA41 = initial_angle.index[::n]
    y_markerA41 = initial_angle.values[::n]

    x_markerA41 = np.concatenate(([initial_angle.index[0]], x_markerA41, [initial_angle.index[-1]]))
    y_markerA41 = np.concatenate(([initial_angle.values[0]], y_markerA41, [initial_angle.values[-1]]))
    data_markerA41.data = dict(x=x_markerA41, y=y_markerA41)

def accept_changesA41(event):
    global coef_A41_GOM
    coef_A41_GOM = dfCoef_A41.copy()
    checked_coefA41 = checkbox_groupA41.selection
    checked_coefA41_2 = [dfCoef_A41.columns.tolist()[row] for row in checked_coefA41]

    for joint_angle in dfCoef_A41.columns.tolist():
        if joint_angle not in checked_coefA41_2:
            coef_A41_GOM[joint_angle] = np.zeros(len(coef_A41_GOM))

def accept_changesA41_KF(event):
    global coef_A41_GOM
    coef_A41_GOM = dfCoef_A41.copy()

## A42

def plot_selected_angleA42(event):
    global angleA42, dfCoef_A42_original, dfCoef_A42, pA42, sourceA42, data_markerA42
    angleA42 = dfCoef_A42.columns.tolist()[event.row]
    selected_coefOrg = dfCoef_A42_original.loc[offset_time:,angleA42].rolling(20, min_periods=1).mean()
    selected_coef = dfCoef_A42.loc[offset_time:,angleA42]

    n = len(selected_coef.index) // 45
    x_markerA42 = selected_coef.index[::n]
    y_markerA42 = selected_coef.rolling(20, min_periods=1).mean().values[::n]
    x_markerA42 = np.concatenate(([selected_coef.index[0]], x_markerA42, [selected_coef.index[-1]]))
    y_markerA42 = np.concatenate(([selected_coef.values[0]], y_markerA42, [selected_coef.values[-1]]))

    pA42.title.text = angleA42
    data_markerA42.data = dict(x=x_markerA42, y=y_markerA42)
    sourceA42.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)

    if compare == True:
        sourceA42_comp.data = dict(x=dfCoef_A42_comp.loc[offset_time:,angleA42].index, y=dfCoef_A42_comp.loc[offset_time:,angleA42].values)

def update_data_marker_changesA42(attr, old, new):
    data_marker_changesA42[angleA42] = {'x': new['x'], 'y': new['y']}

def save_changesA42(event):
    for joint_angle, changes in data_marker_changesA42.items():
        new_values = np.interp(dfCoef_A42.iloc[offset_time:,:].index, changes['x'], changes['y'])
        dfCoef_A42.loc[offset_time:,joint_angle] = pd.Series(index=dfCoef_A42.iloc[offset_time:,:].index, data=new_values)
    #print("Changes A42 saved.")

def save_changesA42_KF(event):
    slider_values = [var.value for var in kf_variables_A42]
    # Update dfCoef_A1 according to the slider values
    dfCoef_A42[:] = slider_values
    #print("Changes A42 saved.")

def reset_changesA42(event):
    for joint_angle in data_marker_changesA42.keys():
        dfCoef_A42.loc[offset_time:,joint_angle] = dfCoef_A42_original.loc[offset_time:,joint_angle]
    #print("Changes A42 reset.")
    
    data_marker_changesA42.clear()
    initial_coef = dfCoef_A42[dfCoef_A42.columns.tolist()[0]].iloc[offset_time:]
    n = len(initial_coef.index) // 45
    x_markerA42 = initial_coef.index[::n]
    y_markerA42 = initial_coef.values[::n]
    x_markerA42 = np.concatenate(([initial_coef.index[0]], x_markerA42, [initial_coef.index[-1]]))
    y_markerA42 = np.concatenate(([initial_coef.values[0]], y_markerA42, [initial_coef.values[-1]]))
    
    pA42.title.text = dfCoef_A42.columns.tolist()[0]
    selected_coefOrg = dfCoef_A42_original[dfCoef_A42.columns.tolist()[0]].iloc[offset_time:].rolling(20, min_periods=1).mean()
    sourceA42.data = dict(x=selected_coefOrg.index, y=selected_coefOrg.values)
    data_markerA42.data = dict(x=x_markerA42, y=y_markerA42)

def reset_changesA42_KF(event):
    for var in kf_variables_A42:
        var._reset()
    #print("Changes A42 reset.")

def reset_jointA42(event):
    dfCoef_A42.loc[offset_time:,angleA42] = dfCoef_A42_original[angleA42].iloc[offset_time:]
    #print(f"Changes to {angleA42} reset.")
    data_marker_changesA42.pop(angleA42, None)
    initial_angle = dfCoef_A42[angleA42].iloc[offset_time:]
    n = len(initial_angle.index) // 45
    x_markerA42 = initial_angle.index[::n]
    y_markerA42 = initial_angle.values[::n]

    x_markerA42 = np.concatenate(([initial_angle.index[0]], x_markerA42, [initial_angle.index[-1]]))
    y_markerA42 = np.concatenate(([initial_angle.values[0]], y_markerA42, [initial_angle.values[-1]]))
    data_markerA42.data = dict(x=x_markerA42, y=y_markerA42)

def accept_changesA42(event):
    global coef_A42_GOM
    coef_A42_GOM = dfCoef_A42.copy()
    checked_coefA42 = checkbox_groupA42.selection
    checked_coefA42_2 = [dfCoef_A42.columns.tolist()[row] for row in checked_coefA42]

    for joint_angle in dfCoef_A42.columns.tolist():
        if joint_angle not in checked_coefA42_2:
            coef_A42_GOM[joint_angle] = np.zeros(len(coef_A42_GOM))

def accept_changesA42_KF(event):
    global coef_A42_GOM
    coef_A42_GOM = dfCoef_A42.copy()

############################################################################################################
# The following functions are used to predict the angles and update the plots and skeleton in the GOM card
# We use the modified coefficients and joint angles to generate the skeleton poses and trajectory
def predict_angle(event):
    global pred_mod, positionsPred, download_Gdat_button
    #print('Predicting angles...')

    if toggle_Model.value != 'KF-GOM':
        coef_A = pd.concat([coef_A1_GOM, coef_A2_GOM, coef_A3_GOM, coef_A41_GOM, coef_A42_GOM], axis=1)
        coef_A = coef_A[col_coef].reset_index(drop=True)

        modified_col = coef_A.columns.tolist()
        coef_mod = coef.copy()
        coef_mod[modified_col] = coef_A[modified_col]
    else:
        if coef_A3_GOM.empty:
            coef_iA = pd.concat([coef_A1_GOM, coef_A2_GOM, coef_A41_GOM, coef_A42_GOM], axis=0)
        else:
            coef_iA = pd.concat([coef_A1_GOM, coef_A2_GOM, coef_A3_GOM, coef_A41_GOM, coef_A42_GOM], axis=0)
        
        coef_iA = coef_iA[row_coef]
        coef_mod = coef.copy()
        coef_mod.loc[:, selectAngle.value+'_'+selectAngleCoef.value+'rotation'] = coef_iA

    if toggle_Model.value == 'ATT-RGOM':
        pred_mod =  ae_gom.pred_ang_coef(eulerAngles_GOM, coef_mod)
    elif toggle_Model.value == 'VAE-RGOM':
        pred_mod =  vae_gom.pred_ang_coef(eulerAngles_GOM, coef_mod)
    elif toggle_Model.value == 'T-RGOM': 
        pred_mod =  t_gom.pred_ang_coef(eulerAngles_GOM, coef_mod)
    elif toggle_Model.value == 'KF-GOM': 
        pred_mod =  kf_gom.pred_ang_coef(eulerAngles_GOM, coef_mod)

    pred_mod = pred_mod - offsetP

    sio = StringIO()
    pred_mod.to_csv(sio)
    sio.seek(0)
    download_Gdat_button = pn.widgets.FileDownload(sio, embed=True, label="Download Generated Data", button_style='solid', width=200, filename='GOM_prediction.csv')


    # Update plot
    fig_pred = go.Figure(
        layout=go.Layout(
            autosize=False,
            width=600,
            height=400
        )
    )
    fig_pred.add_trace(go.Scatter(
            x=np.arange(0, len(pred.iloc[offset_time+3:,0])), 
            y=pred.loc[offset_time+3:, selectAngle.value+'_'+selectAngleCoef.value+'rotation'].rolling(20, min_periods=1).mean(), 
            mode='lines', name='Prediction', line=dict(dash='dash', color='blue')))
    fig_pred.add_trace(go.Scatter(
            x=np.arange(0, len(eulerAngles.iloc[:len(pred.iloc[offset_time+3:,0]),0])), 
            y=eulerAngles.loc[:len(pred.iloc[offset_time+3:,0]), selectAngle.value+'_'+selectAngleCoef.value+'rotation'], 
            mode='lines', name='Original', line=dict(dash='dash', color='gray')))
    fig_pred.add_trace(go.Scatter(
            x=np.arange(0, len(pred_mod.iloc[offset_time+3:,0])), 
            y=pred_mod.loc[offset_time+3:, selectAngle.value+'_'+selectAngleCoef.value+'rotation'].rolling(20, min_periods=1).mean(), 
            mode='lines', name='Modified prediction', line=dict(color='red')))
    fig_pred.update_xaxes(title_text='Time frame')
    fig_pred.update_yaxes(title_text='Angle value')
    fig_pred.update_layout(title={'text': selectAngle.value+'_'+selectAngleCoef.value+'rotation', 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
    fig_pred_plot.object = fig_pred

    # Update skeleton and trajectory
    parsed_dataPred = parsed_data
    dfVal = parsed_dataPred.values
    nn = [dfVal.columns.get_loc(c) for c in pred_mod.columns if c in dfVal]
    first_val = dfVal.iloc[0:3,nn]
    df_val = pd.concat([first_val, pred_mod]).reset_index(drop = True)
    parsed_dataPred.values.iloc[:,nn] = df_val.iloc[:-1]
    positionsPred = mp.fit_transform([parsed_dataPred])

    camA = figA_pred_pane.object.layout['scene']['camera']
    fig_pred_skel = draw_stickfigure3d_js(positionsPred[0], frame=0, cam = camA, highlight_joint = selectAngle.value, GOM_Skel=GOM_Skel)

    trajX = positionsPred[0].values.loc[:,selectAngle.value+'_Xposition'] - positionsPred[0].values.loc[:,'Hips_Xposition']
    trajY = positionsPred[0].values.loc[:,selectAngle.value+'_Zposition'] - positionsPred[0].values.loc[:,'Hips_Zposition']
    trajZ = positionsPred[0].values.loc[:,selectAngle.value+'_Yposition'] - positionsPred[0].values.loc[:,'Hips_Yposition'] + positionsPred[0].values['RightFootToe_Yposition'][0]
    
    fig_pred_skel.add_trace(go.Scatter3d(x=trajX.iloc[offset_time:], y=trajY.iloc[offset_time:],z=trajZ.iloc[offset_time:], mode='lines', 
                                 name='Predicted Trajectory'))
    
    trajX = positions[0].values.loc[:,selectAngle.value+'_Xposition'] - positions[0].values.loc[:,'Hips_Xposition']
    trajY = positions[0].values.loc[:,selectAngle.value+'_Zposition'] - positions[0].values.loc[:,'Hips_Zposition']
    trajZ = positions[0].values.loc[:,selectAngle.value+'_Yposition'] - positions[0].values.loc[:,'Hips_Yposition'] + positions[0].values['RightFootToe_Yposition'][0]
    
    
    fig_pred_skel.add_trace(go.Scatter3d(x=trajX.iloc[offset_time:], y=trajY.iloc[offset_time:],z=trajZ.iloc[offset_time:], mode='lines', 
                                 name='Original Trajectory', line=dict(color='darkgray')))

    fig_pred_skel.update_layout(autosize=False, width=400, height=400)
    figA_pred_pane.object = fig_pred_skel
    figA_pred_pane.param.trigger('object')

    if toggle_Model.value != 'KF-GOM':
        tab_stats = coef_A.describe().T
        tab_stats = tab_stats.reset_index()
        tab_stats = tab_stats.rename(columns={'index': 'Joint Angle'})
        stats_figT.value = tab_stats
        #print('Updated plot and table')
    else:
        tab_stats = pd.concat([coef_iA, dfPvalues], axis=1)
        tab_stats.columns = ['Coefficients', 'P-values']
        stats_figT.value = tab_stats


# Functions for the checkbox of dexterity analysis, according if it is checked or not
def show_cardGOM(checkbox_value, selectAngle_value, angMod_value, toggle_value):
    global variablesOpt, GOM_Skel
    if checkbox_value:                
        selectAngle.options = variablesOpt
        GOM_Skel = True
        camA = figA_pane.object.layout['scene']['camera']
        figA_pane.object = draw_stickfigure3d_js(positions[0], frame=frame_slider.value, cam=camA, highlight_joint = selectAngle.value, GOM_Skel=GOM_Skel, compare=compare)
        figA_pane.param.trigger('object')
        return plot_iGOM(selectAngle_value, angMod_value, toggle_value)
    else:
        selectAngle.options = joints_to_draw
        GOM_Skel = False
        return None

# The following to functions are used to load a new file and update the parsed data (either use ready files or upload a new one)
def select_InitialFile(event):
    global parsed_data, file_name
    file_name = event.new + '.bvh'
    pathComp='bvh2/'+event.new+'.bvh'
    parser = BVHParser()
    parsed_data = parser.parse(pathComp)
    
def upload_InitialFile(event):
    global parsed_data, file_name
    file_name = uploadFile.filename 
    parser = BVHParser()
    parsed_data = parser.parse_input(event.new)

selectFile.param.watch(select_InitialFile, 'value')
uploadFile.param.watch(upload_InitialFile, 'value')
button_file.on_click(load_NewFile)

# We connect our functions to the widgets
icallGOM = pn.bind(show_cardGOM, checkbox_value=checkboxGOM, selectAngle_value=selectAngle, angMod_value=selectAngleCoef, toggle_value=toggle_Model)

# Visualization of our 2D and 3D plots with their widgets (initial main plots)
card2D = pn.Card(tabs, title='2D trajectory', height=560)
centered_fig3D = pn.layout.Column(fig3D_pane, pn.bind(show_intervSlider, select_value=selectInterval), align='center')
card3D = pn.Card(centered_fig3D, title='3D trajectory', height=560)

############################################################################################################
# Functions for comparison with a second file
# The following function is to get the results of GOM on a second comparison file and update the plots and statistics
def plot_iGOM_comp():
    global dfCoef_A1_comp, dfCoef_A2_comp, dfCoef_A3_comp, dfCoef_A41_comp, dfCoef_A42_comp
    global tabsC, sourceA1_comp, sourceA2_comp, sourceA3_comp, sourceA41_comp, sourceA42_comp

    source_comp.data = dict(x=eulerAngles_comp.loc[offset_time:,variables[0]].index, y=eulerAngles_comp.loc[offset_time:,variables[0]].values)

    if toggle_Model.value == 'ATT-RGOM':
        coef_comp, pred_comp = ae_gom.do_gom(eulerAngles_comp)
        #print('ATT-RGOM')
    elif toggle_Model.value == 'VAE-RGOM':
        coef_comp, pred_comp = vae_gom.do_gom(eulerAngles_comp)
        #print('VAE-RGOM')
    elif toggle_Model.value == 'T-RGOM': 
        coef_comp, pred_comp = t_gom.do_gom(eulerAngles_comp)
        #print('T-RGOM')
    elif toggle_Model.value == 'KF-GOM': 
        coef_comp, pred_comp = kf_gom.do_gom(eulerAngles_comp)
        #print('KF-GOM')

    index = variables.index(selectAngle.value+'_'+selectAngleCoef.value+'rotation') #Select joint angle modeled
    
    if toggle_Model.value != 'KF-GOM':
        j_var = varSub[index]
        dfCoef_comp = coef_comp.filter(like=j_var).reset_index(drop=True)
        columns_t3 = dfCoef_comp.filter(like='t-3').columns
        dfCoef_comp = dfCoef_comp.drop(columns=columns_t3)

        col_coef = dfCoef_comp.columns.tolist()
        tab_stats = dfCoef_comp.describe().T
        tab_stats = tab_stats.reset_index()
        tab_stats = tab_stats.rename(columns={'index': 'Joint Angle'})
        stats_figT = pn.widgets.Tabulator(tab_stats, height=300)
    else:
        dfCoef_comp = coef_comp.iloc[:,index]
        dfPvalues_comp = kf_gom.pvalues.iloc[:,index]
        col_coef = coef_comp.columns.tolist()
        row_coef = coef_comp.index.tolist()
        tab_stats = pd.concat([dfCoef_comp, dfPvalues_comp], axis=1)
        tab_stats.columns = ['Coefficients', 'P-values']
        stats_figT = pn.widgets.Tabulator(tab_stats, height=300)
    
    model_title = pn.Row(pn.pane.Markdown('### Second File: '+selectAngle.value+' '+selectAngleCoef.value+'-axis'))

    # Global Statistics
    col_stats_comp = pn.Column(model_title, stats_figT)
    col_stats2 = pn.Row(
        col_stats,
        col_stats_comp,
        width=1050, css_classes=['scrollable']
    )

    pn.config.raw_css = ['.scrollable {overflow-x: auto;}']

    tabsC[5] = ('All assumptions statistics', col_stats2)

    if toggle_Model.value != 'KF-GOM':
 
        # Assumption 1: Time-dependent transition
        pA1_comp = plot_paneA1.object
        dfCoef_A1_comp = dfCoef_comp.filter(regex=selectAngle.value+'_'+selectAngleCoef.value+'rotation')
        sourceA1_comp = ColumnDataSource(data=dict(x=dfCoef_A1_comp.iloc[offset_time:,0].index, y=dfCoef_A1_comp.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA1_comp.line('x', 'y', source=sourceA1_comp, line_width=2, color='lightgrey', line_dash='dashed')
        plot_paneA1.object = pA1_comp


        # Assumption 2: Intra-joint association
        pA2_comp = plot_paneA2.object
        dfCoef_A2_comp = dfCoef_comp.filter(regex=selectAngle.value)
        columns_to_drop = dfCoef_A2_comp.filter(regex=selectAngleCoef.value).columns
        dfCoef_A2_comp = dfCoef_A2_comp.drop(columns=columns_to_drop)
        sourceA2_comp = ColumnDataSource(data=dict(x=dfCoef_A2_comp.iloc[offset_time:,0].index, y=dfCoef_A2_comp.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA2_comp.line('x', 'y', source=sourceA2_comp, line_width=2, color='lightgrey', line_dash='dashed')
        plot_paneA2.object = pA2_comp

        # Assumption 3: Inter-limb synergies
        if selectAngle.value not in nosynergy: 
            if selectAngle.value.startswith('Right'):
                # Do something if selectAngle starts with 'Right'
                selectAngleC = selectAngle.value.replace('Right', 'Left')
            elif selectAngle.value.startswith('Left'):
                selectAngleC = selectAngle.value.replace('Left', 'Right')

            pA3_comp = plot_paneA3.object
            dfCoef_A3_comp = dfCoef_comp.filter(regex=selectAngleC)
            sourceA3_comp = ColumnDataSource(data=dict(x=dfCoef_A3_comp.iloc[offset_time:,0].index, y=dfCoef_A3_comp.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
            pA3_comp.line('x', 'y', source=sourceA3_comp, line_width=2, color='lightgrey', line_dash='dashed')
            plot_paneA3.object = pA3_comp
        else:
            tabA3_comp = pn.Row()
            dfCoef_A3_comp = pd.DataFrame()
            sourceA3_comp = ColumnDataSource(data=dict(x=[], y=[]))

        # Assumption 4.1: Serial Intra-limb mediation
        if positions[0].skeleton[selectAngle.value]['children'] == None:
                listA41 = positions[0].skeleton[selectAngle.value]['parent'] 
        elif positions[0].skeleton[selectAngle.value]['parent'] == None:
                listA41 = positions[0].skeleton[selectAngle.value]['children']
        else:
            listA41 = positions[0].skeleton[selectAngle.value]['children'] + [positions[0].skeleton[selectAngle.value]['parent']]
        
        pA41_comp = plot_paneA41.object
        dfCoef_A41_comp = dfCoef_comp[dfCoef_comp.columns[dfCoef_comp.columns.to_series().str.contains('|'.join(listA41))]]
        sourceA41_comp = ColumnDataSource(data=dict(x=dfCoef_A41_comp.iloc[offset_time:,0].index, y=dfCoef_A41_comp.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA41_comp.line('x', 'y', source=sourceA41_comp, line_width=2, color='lightgrey', line_dash='dashed')
        plot_paneA41.object = pA41_comp

        # Assumption 4.2: Non-Serial Intra-limb mediation
        listA42 = dfCoef_A1_comp.columns.tolist()+ dfCoef_A2_comp.columns.tolist() + dfCoef_A3_comp.columns.tolist()+ dfCoef_A41_comp.columns.tolist()
        
        pA42_comp = plot_paneA42.object
        dfCoef_A42_comp = dfCoef_comp.drop(columns=listA42)
        sourceA42_comp = ColumnDataSource(data=dict(x=dfCoef_A42_comp.iloc[offset_time:,0].index, y=dfCoef_A42_comp.iloc[offset_time:,0].rolling(20, min_periods=1).mean().values))
        pA42_comp.line('x', 'y', source=sourceA42_comp, line_width=2, color='lightgrey', line_dash='dashed')
        plot_paneA42.object = pA42_comp

    #%% Prediction
    fig_pred = fig_pred_plot.object
    fig_pred.add_trace(go.Scatter(
            x=np.arange(0, len(pred_comp.iloc[offset_time+3:,0])), 
            y=pred_comp.loc[offset_time+3:, selectAngle.value+'_'+selectAngleCoef.value+'rotation'].rolling(20, min_periods=1).mean(), 
            mode='lines', name='2nd Recording', line=dict(color='lightgrey', dash='dash')))
    fig_pred_plot.object = fig_pred

    #%% Skeleton
    fig_pred_skel = figA_pred_pane.object

    parsed_dataPred_comp = parsed_data_comp
    dfVal = parsed_dataPred_comp.values
    nn = [dfVal.columns.get_loc(c) for c in pred_comp.columns if c in dfVal]
    first_val = dfVal.iloc[0:3,nn]
    df_val = pd.concat([first_val, pred_comp]).reset_index(drop = True)
    parsed_dataPred_comp.values.iloc[:,nn] = df_val
    positionsPred_comp = mp.fit_transform([parsed_dataPred_comp])

    trajX = positionsPred_comp[0].values.loc[:,selectAngle.value+'_Xposition'] - positionsPred_comp[0].values.loc[:,'Hips_Xposition']
    trajY = positionsPred_comp[0].values.loc[:,selectAngle.value+'_Zposition'] - positionsPred_comp[0].values.loc[:,'Hips_Zposition']
    trajZ = positionsPred_comp[0].values.loc[:,selectAngle.value+'_Yposition'] - positionsPred_comp[0].values.loc[:,'Hips_Yposition'] + positionsPred_comp[0].values['RightFootToe_Yposition'][0]
    
    fig_pred_skel.add_trace(go.Scatter3d(x=trajX.iloc[offset_time:], y=trajY.iloc[offset_time:],z=trajZ.iloc[offset_time:], mode='lines', 
                                 name='2nd Recording', line=dict(color='green')))
    
    fig_pred_skel.update_layout(autosize=False, width=400, height=400)
    figA_pred_pane.object = fig_pred_skel
    
# Function to load a new file for comparison
def load_NewCompareFile(event):
    global compare, eulerAngles_comp, jpos_comp, positions_comp
    compare = True

    positions_comp = mp.fit_transform([parsed_data_comp])
    mocap_data = parsed_data_comp.values.reset_index().iloc[:,1:]
    rot_cols = [col for col in mocap_data.columns if 'rotation' in col]
    eulerAngles_comp = mocap_data[rot_cols]
    jpos_comp = positions_comp[0].values.iloc[1:,:].reset_index(drop=True)


    fig2Dx = fig2Dx_pane.object
    fig2Dy = fig2Dy_pane.object
    fig2Dz = fig2Dz_pane.object
    fig3D = fig3D_pane.object

    frame = frame_slider.value
    if frame >= len(eulerAngles_comp)-offset_time:
        frame = len(eulerAngles_comp)-offset_time-1

    dff_3d = eulerAngles_comp[[selectAngle.value+'_Xrotation', selectAngle.value+'_Yrotation', selectAngle.value+'_Zrotation']]

    dat_x = dff_3d.iloc[offset_time:,0].values
    dat_y = dff_3d.iloc[offset_time:,1].values
    dat_z = dff_3d.iloc[offset_time:,2].values   

    fig2Dx.add_trace(
        go.Scatter(
            name=dff_3d.columns[0] + ' (2nd Recording)',
            x=np.arange(0, len(dat_x)),
            y=dat_x,
            mode='lines'
        ))
    fig2Dx.add_trace(
        go.Scatter(
            x=[frame],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dat_x[frame]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
    ))

    fig2Dy.add_trace(
        go.Scatter(
            name=dff_3d.columns[1]+' (2nd Recording)',
            x=np.arange(0, len(dat_y)),
            y=dat_y,
            mode='lines'
        ))
    fig2Dy.add_trace(
        go.Scatter(
            x=[frame],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dat_y[frame]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
    ))

    fig2Dz.add_trace(
        go.Scatter(
            name=dff_3d.columns[2]+' (2nd Recording)',
            x=np.arange(0, len(dat_z)),
            y=dat_z,
            mode='lines'
        ))
    fig2Dz.add_trace(
        go.Scatter(
            x=[frame],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dat_z[frame]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
    ))
    
    fig3D.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], mode='lines', 
                                 name=selectAngle.value+' (2nd Recording)'))
    fig3D.add_trace(go.Scatter3d(x=[dff_3d.iloc[frame,0]], y=[dff_3d.iloc[frame,1]], 
                                 z=[dff_3d.iloc[frame,2]], mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))
    
    fig2Dx_pane.object = fig2Dx
    fig2Dy_pane.object = fig2Dy
    fig2Dz_pane.object = fig2Dz
    fig3D_pane.object = fig3D
    figA_pane.object = draw_stickfigure3d_js(positions[0], frame=frame_slider.value, highlight_joint = selectAngle.value, GOM_Skel=GOM_Skel, compare=compare)

    if checkboxGOM.value:
        #source_comp.data = dict(x=eulerAngles_comp[jointA].iloc[offset_time:].index, y=eulerAngles_comp[jointA].iloc[offset_time:].values)
        plot_iGOM_comp()



# Functions to select and upload a new file for comparison
def select_Comparefile(event):
    if event.new:
        inputCompareFile.disabled = True

        global parsed_data_comp
        pathComp='bvh2/'+event.new+'.bvh'
        parser = BVHParser()
        parsed_data_comp = parser.parse(pathComp)

    else:
        inputCompareFile.disabled = False
    
def upload_Comparefile(event):
    if event.new:
        selectCompareFile.disabled = True

        global parsed_data_comp
        parser = BVHParser()
        parsed_data_comp = parser.parse_input(event.new)
    else:
        selectCompareFile.disabled = False

buttonCompare.on_click(load_NewCompareFile)

# Function to remove the comparison with a second file
def remove_comparison(event):
    global compare
    compare = False
    inputCompareFile.disabled = False
    selectCompareFile.disabled = False

    fig3D = go.Figure(layout=dict(width=690, height=450))
    fig2Dx = go.Figure(layout=dict(width=690, height=450))
    fig2Dy = go.Figure(layout=dict(width=690, height=450))
    fig2Dz= go.Figure(layout=dict(width=690, height=450))

    dff_3d = eulerAngles[[selectAngle.value+'_Xrotation', selectAngle.value+'_Yrotation', selectAngle.value+'_Zrotation']]

    dat_x = dff_3d.iloc[offset_time:,0].values
    dat_y = dff_3d.iloc[offset_time:,1].values
    dat_z = dff_3d.iloc[offset_time:,2].values   

    fig2Dx.add_trace(
        go.Scatter(
            name=dff_3d.columns[0],
            x=np.arange(0, len(dat_x)),
            y=dat_x,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dx.add_trace(
        go.Scatter(
            x=[frame_slider.value],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dat_x[frame_slider.value]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(255, 0, 0, 0.5)')
    ))
    fig2Dx.update_xaxes(title_text='Time frame')
    fig2Dx.update_yaxes(title_text='Angle (degrees)')

    fig2Dy.add_trace(
        go.Scatter(
            name=dff_3d.columns[1],
            x=np.arange(0, len(dat_y)),
            y=dat_y,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dy.add_trace(
        go.Scatter(
            x=[frame_slider.value],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dat_y[frame_slider.value]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 128, 0, 0.5)')
    ))
    fig2Dy.update_xaxes(title_text='Time frame')
    fig2Dy.update_yaxes(title_text='Angle (degrees)')

    fig2Dz.add_trace(
        go.Scatter(
            name=dff_3d.columns[2],
            x=np.arange(0, len(dat_z)),
            y=dat_z,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ))
    fig2Dz.add_trace(
        go.Scatter(
            x=[frame_slider.value],  # Replace with the x-coordinate of the marker in the initial frame
            y=[dat_z[frame_slider.value]],  # Replace with the y-coordinate of the marker in the initial frame
            mode='markers',
            showlegend=False,
            marker=dict(size=10, symbol="diamond-dot", color='rgba(0, 0, 255, 0.5)')
    ))
    fig2Dz.update_xaxes(title_text='Time frame')
    fig2Dz.update_yaxes(title_text='Angle (degrees)')


    cam3D = fig3D_pane.object.layout['scene']['camera']
    fig3D.add_trace(go.Scatter3d(x=dff_3d.iloc[offset_time:,0], y=dff_3d.iloc[offset_time:,1],z=dff_3d.iloc[offset_time:,2], 
                                 mode='lines', name=selectAngle.value))
    fig3D.add_trace(go.Scatter3d(x=[dff_3d.iloc[frame_slider.value,0]], y=[dff_3d.iloc[frame_slider.value,1]], z=[dff_3d.iloc[frame_slider.value,2]], 
                                 mode='markers', showlegend=False, marker=dict(size=5, symbol='diamond')))

    fig2Dx_pane.object = fig2Dx
    fig2Dy_pane.object = fig2Dy
    fig2Dz_pane.object = fig2Dz

    fig3D.layout['scene']['camera'] = cam3D
    fig3D_pane.object = fig3D
    camA = figA_pane.object.layout['scene']['camera']
    figA_pane.object = draw_stickfigure3d_js(positions[0], frame=frame_slider.value, cam=camA, highlight_joint = selectAngle.value, GOM_Skel=GOM_Skel, compare=compare)
    
    if checkboxGOM.value:
        tabsC[5] = ('All assumptions statistics', col_stats)

        if toggle_Model.value != 'KF-GOM':
            source_comp.data = dict(x=[], y=[])
            sourceA1_comp.data = dict(x=[], y=[])
            sourceA2_comp.data = dict(x=[], y=[])
            sourceA3_comp.data = dict(x=[], y=[])
            sourceA41_comp.data = dict(x=[], y=[])
            sourceA42_comp.data = dict(x=[], y=[])
            
            plot_paneA1.object.legend.visible = False
            plot_paneA2.object.legend.visible = False
            plot_paneA41.object.legend.visible = False
            plot_paneA42.object.legend.visible = False
            if selectAngle.value not in nosynergy: 
                plot_paneA3.object.legend.visible = False

        fig_pred = fig_pred_plot.object
        new_data = [trace for trace in fig_pred.data if trace.name != '2nd Recording']
        fig_pred.data = new_data
        fig_pred_plot.object = fig_pred

        fig_pred_skel = figA_pred_pane.object 
        new_data = [trace for trace in fig_pred_skel.data if trace.name !='2nd Recording']
        fig_pred_skel.data = new_data
        fig_pred_skel.update_layout(autosize=False, width=400, height=400)
        figA_pred_pane.object = fig_pred_skel


    print(f'Compare is now: {compare}')

buttonRemoveCompare.on_click(remove_comparison)

# Add the callback functions to the widgets
selectCompareFile.param.watch(select_Comparefile, 'value')
inputCompareFile.param.watch(upload_Comparefile, 'value')

# Visualization of the widgets to load a second file for comparison in the side bar
file_Compare_selection = pn.Column("#### Load Second Recording for Comparison:",
                                   pn.layout.Divider(margin=(-30, 0, 0, 0)),
                                   "Choose File from Repository or Load New File:",
                                   selectCompareFile, 
                                   inputCompareFile, 
                                   buttonCompare, 
                                   buttonRemoveCompare)

############################################################################################################

# Callback functions for the widgets in the side bar
def show_card2D(checkbox_value):
    return card2D if checkbox_value else None

def show_card3D(checkbox_value):
    return card3D if checkbox_value else None

data_marker.on_change('data', update_data_marker_changes)
checkbox_group.on_click(plot_selected_angle)
save_button.on_click(save_changes)
reset_Jbutton.on_click(reset_joint)
reset_Abutton.on_click(reset_changes)
accept_button.on_click(accept_changes)

# Instantiate the template with widgets displayed in the sidebar
app = pn.template.MaterialTemplate(
    title='AImove',
    sidebar=["#### Load MoCap Recording",
             pn.layout.Divider(margin=(-30, 0, 0, 0)),
             "Select File from Repository or Load New File:",
             selectFile,
             uploadFile,
             button_file,
             "#### Visualization Controls",
             pn.layout.Divider(margin=(-30, 0, 0, 0)),
             selectAngle,
             frame_slider,
             "Type of plots to display:",
             checkbox2D, 
             checkbox3D, 
             "#### Analysis",
             pn.layout.Divider(margin=(-30, 0, 0, 0)),
             checkboxGOM,
             selectInterval,
             checkboxKinematics,
             "#### Load Second Recording for Comparison:",
             pn.layout.Divider(margin=(-30, 0, 0, 0)),
             "Select File from Repository or Load New File:",
             selectCompareFile,
             inputCompareFile,
             buttonCompare, 
             buttonRemoveCompare],
    header_background='#ffffff',  # Set the background color to white
    logo='Logo-Banner-A2-1.png'
)

# Add the main plots, widgets, and functions to the app
app.main.append(
    pn.Row(
        pn.Column(
            pn.Card(figA_pane, title='Animation', height=560),
        ),
        pn.Column(
            pn.bind(show_card2D, checkbox_value=checkbox2D),
            pn.bind(show_card3D, checkbox_value=checkbox3D)),
        pn.Column(icallGOM),
        iupdate_angle,
        iupdate_interval,
    )
)


app.show()
