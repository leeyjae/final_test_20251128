import streamlit as st
import plotly.graph_objects as go
import numpy as np


st.set_page_config(page_title='직육면체 · 정육면체: 블록 쌓기 학습', page_icon=':blue_square:')


def cube_vertices(x, y, z, offset=(0, 0, 0)):
    x0, y0, z0 = offset
    return np.array([
        [x0, y0, z0],
        [x0 + x, y0, z0],
        [x0 + x, y0 + y, z0],
        [x0, y0 + y, z0],
        [x0, y0, z0 + z],
        [x0 + x, y0, z0 + z],
        [x0 + x, y0 + y, z0 + z],
        [x0, y0 + y, z0 + z],
    ])


def faces_from_vertices(offset_index=0):
    v = np.arange(8) + offset_index
    return [
        (v[0], v[1], v[2]), (v[0], v[2], v[3]),
        (v[4], v[6], v[5]), (v[4], v[7], v[6]),
        (v[0], v[4], v[5]), (v[0], v[5], v[1]),
        (v[2], v[6], v[7]), (v[2], v[7], v[3]),
        (v[0], v[3], v[7]), (v[0], v[7], v[4]),
        (v[1], v[5], v[6]), (v[1], v[6], v[2])
    ]


def add_cuboid_mesh(fig, w, d, h, origin=(0, 0, 0), face_colors=None, opacity=0.9, showlegend=False):
    verts = cube_vertices(w, d, h, offset=origin)
    tris = np.array(faces_from_vertices())
    tri_face_type = ['wd', 'wd', 'wh', 'wh', 'wh', 'wh', 'dh', 'dh', 'dh', 'dh', 'dh', 'dh']
    x, y, z = verts.T
    for face_type in ['wd', 'wh', 'dh']:
        mask = [i for i, t in enumerate(tri_face_type) if t == face_type]
        i = tris[mask, 0]
        j = tris[mask, 1]
        k = tris[mask, 2]
        color = face_colors.get(face_type, '#999') if face_colors else '#999'
        # map face_type to label for legend
        label = {
            'wd': '가로×세로 (width×depth)',
            'wh': '가로×높이 (width×height)',
            'dh': '세로×높이 (depth×height)'
        }.get(face_type, face_type)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=label, showlegend=showlegend))


def face_rect_fig(a, b, color, title=None):
    """Create a small 2D Plotly figure that draws an a×b rectangle colored.

    a,b values are used both for sizing and shown in the annotation.
    """
    fig = go.Figure()
    # Draw rectangle as shape
    fig.update_layout(xaxis=dict(range=[0, a + a * 0.2], visible=False),
                      yaxis=dict(range=[0, b + b * 0.2], visible=False),
                      width=220, height=160,
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.add_shape(type='rect', x0=0, y0=0, x1=a, y1=b, fillcolor=color, line=dict(color='black'))
    fig.add_annotation(x=a / 2, y=b / 2, text=f'{a} × {b} = {a * b}', showarrow=False, font=dict(color='white'))
    if title:
        fig.update_layout(title=title)
    return fig


def cube_mesh_for_unit_cubes(w, d, h):
    faces = faces_from_vertices()
    all_x, all_y, all_z = [], [], []
    i_list, j_list, k_list = [], [], []
    tri_offset = 0
    for xi in range(int(w)):
        for yi in range(int(d)):
            for zi in range(int(h)):
                v = cube_vertices(1, 1, 1, offset=(xi, yi, zi))
                x, y, z = v.T
                all_x.extend(x.tolist())
                all_y.extend(y.tolist())
                all_z.extend(z.tolist())
                for t in faces:
                    i_list.append(int(t[0] + tri_offset))
                    j_list.append(int(t[1] + tri_offset))
                    k_list.append(int(t[2] + tri_offset))
                tri_offset += 8
    return go.Mesh3d(x=all_x, y=all_y, z=all_z, i=i_list, j=j_list, k=k_list, color='#a0a0ff', opacity=0.95)


def cube_mesh_for_unit_cubes_layers(w, d, h, upto_layer):
    """Build a mesh containing unit-cubes only up to the given layer (1..h)."""
    faces = faces_from_vertices()
    all_x, all_y, all_z = [], [], []
    i_list, j_list, k_list = [], [], []
    tri_offset = 0
    upto = max(0, min(int(upto_layer), int(h)))
    for xi in range(int(w)):
        for yi in range(int(d)):
            for zi in range(upto):
                v = cube_vertices(1, 1, 1, offset=(xi, yi, zi))
                x, y, z = v.T
                all_x.extend(x.tolist())
                all_y.extend(y.tolist())
                all_z.extend(z.tolist())
                for t in faces:
                    i_list.append(int(t[0] + tri_offset))
                    j_list.append(int(t[1] + tri_offset))
                    k_list.append(int(t[2] + tri_offset))
                tri_offset += 8
    if not all_x:
        # empty mesh: return a very small invisible mesh to avoid plotly errors
        return go.Mesh3d(x=[0], y=[0], z=[0], i=[0], j=[0], k=[0], opacity=0)
    return go.Mesh3d(x=all_x, y=all_y, z=all_z, i=i_list, j=j_list, k=k_list, color='#a0a0ff', opacity=0.95)


# NOTE: removed cube_mesh_for_unit_cubes_colored to revert surface-color grouping changes


st.title('직육면체 · 정육면체: 블록 쌓기로 배우는 부피와 겉넓이')
st.write('입력한 가로, 세로, 높이를 바탕으로 3D 모델을 보여주고 부피와 겉넓이 계산을 색깔로 설명합니다.')

with st.sidebar:
    st.header('설정')
    unit = st.selectbox('단위', ['cm', 'm'], index=0)
    st.markdown('단위는 표시용입니다 — 블록 하나는 1×1×1 (선택한 단위)로 가정합니다.')
    w = st.number_input('가로 (width, 정수)', min_value=1, max_value=40, value=4, step=1)
    d = st.number_input('세로 (depth, 정수)', min_value=1, max_value=40, value=3, step=1)
    h = st.number_input('높이 (height, 정수)', min_value=1, max_value=40, value=2, step=1)
    st.markdown('옵션: 단위 블록으로 쌓아 보여주기 (성능 제한 적용)')
    show_blocks = st.checkbox('단위 블록으로 쌓기 (1×1×1)', value=True)
    max_render = st.slider('블록 렌더 최대 개수', min_value=64, max_value=8192, value=512, step=64)

volume = int(w) * int(d) * int(h)
base_area = int(w) * int(d)

st.subheader('부피 (Volume)')
st.write(f'공식: 가로 × 세로 × 높이 → {w} × {d} × {h} (단위: {unit})')
st.metric('부피', f'{volume} {unit}³')
st.write(f'총 블록 수 = **{volume}** 개 — 블록 하나는 1×1×1 ({unit})')

# Visual explanation for volume
col_v1, col_v2 = st.columns([1, 2])
with col_v1:
    base_fig = face_rect_fig(int(w), int(d), '#a08fda', title='밑면: 가로×세로')
    st.plotly_chart(base_fig, use_container_width=True)
with col_v2:
    st.markdown('**부피 계산 과정 (시각적 단계)**')
    st.write(f'1) 밑면 넓이 = 가로 × 세로 = {w} × {d} = **{base_area} {unit}²**')
    st.write(f'2) 부피 = 밑면 넓이 × 높이 = {base_area} × {h} = **{volume} {unit}³**')
    st.write('→ 이 값은 실제로 쌓인 단위 블록(1×1×1)의 총 개수와 동일합니다.')

st.subheader('겉넓이 (Surface area) — 색깔로 과정을 보여줍니다')
a_wd = int(w) * int(d)
a_wh = int(w) * int(h)
a_dh = int(d) * int(h)
total_sa = 2 * (a_wd + a_wh + a_dh)

colors = {'wd': '#ff6666', 'wh': '#66cc66', 'dh': '#66a3ff'}
colA, colB, colC = st.columns(3)
with colA:
    st.markdown(f"<div style='background:{colors['wd']};padding:8px;border-radius:6px;color:#fff'>가로×세로 = {a_wd} {unit}²</div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<div style='background:{colors['wh']};padding:8px;border-radius:6px;color:#fff'>가로×높이 = {a_wh} {unit}²</div>", unsafe_allow_html=True)
with colC:
    st.markdown(f"<div style='background:{colors['dh']};padding:8px;border-radius:6px;color:#fff'>세로×높이 = {a_dh} {unit}²</div>", unsafe_allow_html=True)

# Show small 2D rectangles for each face so calculation is visual
fig1 = face_rect_fig(int(w), int(d), colors['wd'], title='가로×세로')
fig2 = face_rect_fig(int(w), int(h), colors['wh'], title='가로×높이')
fig3 = face_rect_fig(int(d), int(h), colors['dh'], title='세로×높이')

viz_col1, viz_col2, viz_col3 = st.columns(3)
viz_col1.plotly_chart(fig1, use_container_width=True)
viz_col2.plotly_chart(fig2, use_container_width=True)
viz_col3.plotly_chart(fig3, use_container_width=True)

# step-by-step surface area calculation
st.markdown('**겉넓이 계산 과정 (시각적 단계)**')
st.write(f'1) 서로 다른 면의 넓이를 계산한다: 가로×세로={a_wd}, 가로×높이={a_wh}, 세로×높이={a_dh}')
st.write(f'2) 위 세 넓이의 합 = {a_wd} + {a_wh} + {a_dh} = **{a_wd + a_wh + a_dh} {unit}²**')
st.write(f'3) 최종 겉넓이 = 2 × (합) = 2 × {a_wd + a_wh + a_dh} = **{total_sa} {unit}²**')

st.write('겉넓이 계산: 2 × (가로×세로 + 가로×높이 + 세로×높이)')
st.metric('겉넓이', f'{total_sa} {unit}²')

st.subheader('3D 시각화')
fig = go.Figure()
add_cuboid_mesh(fig, w, d, h, origin=(0, 0, 0), face_colors=colors, opacity=0.5)

if show_blocks:
    if volume <= max_render:
        # show a layer slider so learners can build the cuboid layer by layer
        layers_to_show = st.slider('쌓을 레이어 수 (0 = 외피만)', min_value=0, max_value=int(h), value=int(h))
        # produce colored surface-blocks for the shown range
        if layers_to_show >= int(h):
            mesh = cube_mesh_for_unit_cubes(w, d, h)
            fig.add_trace(mesh)
        else:
            # for partial layers, color only blocks up to layer
            partial = cube_mesh_for_unit_cubes_layers(w, d, h, upto_layer=layers_to_show)
            fig.add_trace(partial)
        st.caption('레이어 슬라이더로 블록이 위로 쌓이는 과정을 확인하세요.')
    else:
        st.info(f'총 {volume}개의 블록은 렌더링하기에 너무 많습니다. 렌더 임계값: {max_render}개로 설정되어 있습니다.')

fig.update_layout(scene=dict(
    xaxis=dict(title='width', visible=True),
    yaxis=dict(title='depth', visible=True),
    zaxis=dict(title='height', visible=True),
    aspectmode='data'
), margin=dict(l=0, r=0, t=0, b=0), height=640)

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')
st.write('앱 설명: 위 모델에서 각 서로다른 면의 넓이를 색깔로 구분해 보여주며(가로×세로 / 가로×높이 / 세로×높이)\\n겉넓이는 이 3개의 넓이를 더한 뒤 2를 곱해 구합니다. 부피는 가로×세로×높이, 즉 단위 블록의 총 개수와 같습니다.')
