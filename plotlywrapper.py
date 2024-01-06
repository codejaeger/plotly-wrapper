import itertools
import numpy as np

from plotly.subplots import make_subplots
from plotly.colors import label_rgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def func_xyz2uv():
    def xyz2uv(v, icam):
        """Convert from XYZ to UVD coordinate frame using the camera coordinates.
        
        Args:
        v - Tensor, list of vertices.
        icam - intrinsic camera coordinates.
        """
        fx, fy, cx, cy = icam[0, 0], icam[1, 1], icam[0, 2], icam[1, 2]
        x_d = v[:, 0] / v[:, 2] * fx + cx
        y_d = v[:, 1] / v[:, 2] * fy + cy
        print(np.concatenate([x_d, y_d], axis=-1).shape)
        print(x_d, y_d, v.shape, np.concatenate([x_d, y_d], axis=-1).shape)
        return np.concatenate([x_d[:, np.newaxis], y_d[:, np.newaxis]], axis=-1)
    return xyz2uv

class BasePlotter:
    """
    Produce 2D/3D graphics plots.
    
    This is a wrapper class around the Plotly provided graph objects to ease the arrangement 
    of figures in a tabular manner. 
    
    This class contains contains many utility functions to simplify using plotly and the 
    method names easily convey the purpose of the functions. 
    """
    def __init__(self, gobjs=None, specs=None, attrs=None, **kwargs):
        """
        Args:
        gobjs - The graph objects i.e. plotly.graph_objects.
        specs - Specifications for the graph object i.e. whether the object
        is considered 2D or 3D.
        attrs - Additional attributes for the graph objects for example,
        whether to draw a colormap for the graph object.
        
        Accepted input formats for gobjs:
        (gobj) - single plot,
        (gobjs) - single plot,
        [(gobj/s) or None, (gobj/s) or None] - row plot,
        [[(gobj/s) or None, (gobj/s) or None], [(gobj/s) or None, (gobj/s) or None]] - table plot
        
        Accepted input formats for specs:
        [[{"type":}]]
        [[{"type":} or None, {"type":} or None]]
        [[{"type":} or None, {"type":} or None], [{"type":} or None, {"type":} or None]]
        
        Accepted input formats for attrs:
        [[{'attr': val}]]
        [[{'attr': val}, {'attr': val}]]
        [[{'attr': val}, {'attr': val}], [{'attr': val}, {'attr': val}]]
        """
        self.gobjs = gobjs
        self.specs = specs
        self.attrs = attrs
        self.kw = {**{'horizontal_spacing': 0.1, 'vertical_spacing': 0.1, 'shared_xaxes': 'columns', 'shared_yaxes': 'rows'} , **kwargs}
        self.fig = self.make_subplot()
    
    def make_subplot(self):
        # Check if it is a single plot type.
        if isinstance(self.gobjs, tuple):
            self.rows, self.cols = 1, 1
            fig = make_subplots(rows=1, cols=1, specs=self.specs, **self.kw)
            for i in self.gobjs:
                fig.add_trace(i, 1, 1)
                
        # Check if it is a horizontal subplot type.
        elif isinstance(self.gobjs, list) and (isinstance(self.gobjs[0], tuple) or self.gobjs[0]==None):
            self.rows, self.cols = 1, len(self.gobjs)
            fig = make_subplots(rows=1, cols=self.cols, specs=self.specs, **self.kw)
            if self.cols > 1:
                self.kw['horizontal_spacing'] = 0.1 / (self.cols - 1)
            for idx, objs in enumerate(self.gobjs):
                if objs is None:
                    continue
                for obj in objs:
                    fig.add_trace(obj, 1, idx+1)
                    
                    # adjust spacing between plots to include colorbars
                    if self.attrs[0][idx]['has_colorbar']:
                        gap = 1.1 / self.cols
                        offset = (0.2 ) * gap
                        fig.for_each_trace(lambda trace: trace.update(colorbar=dict(len=1, x=gap*(idx+1)-offset*(1+idx*(1-offset)*0.1), thickness=30/(self.cols))) if trace.type == "heatmap" or trace.type == "surface" else (), row=1, col=idx+1)
                        
        # Check if it is a vertical or a table subplot type.
        elif isinstance(self.gobjs, list) and isinstance(self.gobjs[0], list) and (isinstance(self.gobjs[0][0], tuple) or self.gobjs[0][0]==None):
            self.rows, self.cols = len(self.gobjs), len(self.gobjs[0])
            if self.cols > 1:
                self.kw['horizontal_spacing'] = 0.1 / (self.cols - 1)
            if self.rows > 1:
                self.kw['vertical_spacing'] = 0.1 / (self.rows - 1)
            fig = make_subplots(rows=self.rows, cols=self.cols, specs=self.specs, **self.kw)
            for i, r in enumerate(self.gobjs):
                for j, c in enumerate(r):
                    if c is None:
                        continue
                    for obj in c:
                        fig.add_trace(obj, i+1, j+1)

                        # adjust spacing between plots to include colorbars
                        if self.attrs[i][j]['has_colorbar']:
                            gap_x = 1.1 / self.cols
                            offset_x = (0.2 ) * gap_x 
                            gap_y = 1 / self.rows
                            offset_y = (0.5 ) * gap_y
                            fig.for_each_trace(lambda trace: trace.update(colorbar=dict(x=gap_x*(j+1)-offset_x*(1+j*(1-offset_x)*0.1), y=gap_y*(self.rows - i)-offset_y, len=gap_y, thickness=30/(self.cols))) if trace.type == "heatmap" or trace.type == "surface" else (), row=i+1, col=j+1)
        return fig

    def update_kw(self, **kwargs):
        self.kw = kwargs
        self.fig = self.make_subplot()
        return self
    
    def remove_3d_background(self, rows=None, cols=None):
        for r, c in self.parse_update_indices(rows, cols):
            self.fig.update_scenes(
                    xaxis=dict(showbackground=False, showticklabels=False,
                              title="", visible=False),
                    yaxis=dict(showbackground=False, showticklabels=False,
                              title="", visible=False),
                    zaxis=dict(showbackground=False, showticklabels=False,
                              title="", visible=False),
            row=r, col=c,
            )
        return self
    
    def remove_axis_points(self, rows=None, cols=None):
        for r, c in self.parse_update_indices(rows, cols):
            self.fig.update_layout(showlegend=False)
            self.fig.update_xaxes(visible=False)
            self.fig.update_yaxes(visible=False)
        return self
    
    def remove_legend(self, rows=None, cols=None):
        self.fig.update_layout(showlegend=False)
        for r, c in self.parse_update_indices(rows, cols):
            try:
                self.fig.update_traces(showlegend=False, 
                                       row=r, col=c
                                      )
            except:
                pass
        return self
        
    def remove_colorscale(self, rows=None, cols=None):
        for r, c in self.parse_update_indices(rows, cols):
            self.fig.update_traces(showscale=False, 
                                   row=r, col=c
                                  )
            try:
                self.fig.update_coloraxes(showscale=False, 
                                          row=r, col=c
                                         )
            except:
                pass
        return self

    def remove_2d_grid(self, rows=None, cols=None):
        for r, c in self.parse_update_indices(rows, cols):
            self.fig = self.fig.update_xaxes(showgrid=False, 
                                  row=r, col=c
                                 )
            self.fig = self.fig.update_yaxes(showgrid=False, 
                                  row=r, col=c
                                 )
        return self
    
    def white_pub_grid(self, rows=None, cols=None):
        self.fig.update_layout(
                plot_bgcolor='white'
            )
        for r, c in self.parse_update_indices(rows, cols):
            self.fig = self.fig.update_xaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey', 
                row=r, 
                col=c
            )
            self.fig = self.fig.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey',
                row=r,
                col=c
            )
        return self
    
    def update_size(self, width=320, height=320, l=10, r=10, t=10, b=10, pad=1):
        self.fig.update_layout(
            width=width,
            height=height,
            margin=dict(
                l=l,
                r=r,
                b=b,
                t=t,
                pad=pad
            ), 
        )
        return self
    
    def hide_background(self):
        self.fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return self
    
    def invert_y(self, rows=None, cols=None):
        for r, c in self.parse_update_indices(rows, cols):
            self.fig = self.fig.update_yaxes(autorange='reversed', 
                                             row=r, col=c
                                            )
        return self
   
    def scale_axis_to_same(self, rows=None, cols=None):
        for r, c in self.parse_update_indices(rows, cols):
            self.fig = self.fig.update_yaxes(
                scaleanchor = "x",
                scaleratio = 1,
                row=r, col=c
              )
        return self
        
    def fit_to_shape(self, shape):
        self.fig = self.fig.update_layout(width=shape[1], height=shape[0], 
                                         )
        return self
        
    def update_camera(self, rows=None, cols=None):
        camera = dict(
            up=dict(x=0, y=-1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=-2)
        )
        for r, c in self.parse_update_indices(rows, cols):
            self.fig.update_scenes(
                camera=camera,
                row=r, col=c,
            )
        return self
    
    def parse_update_indices(self, rows=None, cols=None):
        if rows != None and cols != None:
            plotids = zip(rows, cols)
        else:
            rows = np.arange(self.rows)+1 if rows == None else rows
            cols = np.arange(self.cols)+1 if cols == None else cols
            plotids = itertools.product(rows, cols)
        return plotids
    
    def _repr_html_(self):
        self.fig.show()
        
    def save_file(self, path):
        if path[-3:] == 'png':
            self.fig.write_image(path)
        else:            
            self.fig.write_html(path)
        
    
class Plotter(BasePlotter):
    def __init__(self, gobjs=None, specs=None, attrs=None):
        """This a dervied plotter type which can only be a single plot 
        (or a collection of similar subplots merged into a single plot).
        
        Accepted format for gobjs:
        (gobj/s) - a single plot or a tuple of subplots.
        """
        super().__init__(gobjs, specs, attrs)
    
    def update_specs(self, **kwargs):
        self.specs = [[{**self.specs[0][0], **kwargs}]]
        return self
    
    def __add__(self, otherplotter):
        _gobjs = tuple(list(self.gobjs) + list(otherplotter.gobjs)) 
        self.attrs = [[{'has_colorbar': self.attrs[0][0]['has_colorbar'] or otherplotter.attrs[0][0]['has_colorbar']}]]
        return Plotter(gobjs=_gobjs, specs=self.specs, attrs=self.attrs)

def make_plot(p, **kwargs):
    """Use this method to parse the input (list/tuple/table) of subplots before creating
    a BasePlotter object for them.
    
    Args:
    p - a single (or tuple), row or a table (list) of plotly graph objects
    """
    if isinstance(p, Plotter):
        return p.update_kw(**kwargs)
    elif isinstance(p, list) and isinstance(p[0], Plotter):
        gobjs, specs, attrs = [], [], []
        for i in p:
            if i is None:
                gobjs.append(None)
                specs.append(None)
                attrs.append(None)
            else:
                gobjs.append(i.gobjs)
                specs.append(i.specs[0][0])
                attrs.append(i.attrs[0][0])
        return BasePlotter(gobjs=gobjs, specs=[specs], attrs=[attrs], **kwargs)
    elif isinstance(p, list) and isinstance(p[0], list) and isinstance(p[0][0], Plotter):
        gobjs, specs, attrs = [], [], []
        for r in p:
            _gobjs, _specs, _attrs = [], [], []
            for c in r:
                if c is None:
                    _gobjs.append(None)
                    _specs.append(None)
                    _attrs.append(None)
                else:
                    _gobjs.append(c.gobjs)
                    _specs.append(c.specs[0][0])
                    _attrs.append(c.attrs[0][0])
            gobjs.append(_gobjs)
            specs.append(_specs)
            attrs.append(_attrs)
        return BasePlotter(gobjs=gobjs, specs=specs, attrs=attrs, **kwargs)

    
#####################################################
# All inputs should be numpy matrices, arrays etc.  #
#####################################################
# Define all visualisation plotting utility classes #
#####################################################

class Mesh3D(Plotter):
    """Plotting class for 3D meshes.
    
    Args:
    v - vertices
    F - Faces
    hoverinfo - whether hovering over points show additional information
    intr_cam - intrinsic camera
    """
    def __init__(self, v, F, opacity=1, color=None, hoverinfo=None, intr_cam=None, **kwargs):
        self.v, self.F, self.opacity, self.color, self.intr_cam = v, F, opacity, color, intr_cam
        self.hoverinfo = hoverinfo
        self.kwargs = kwargs
        
        self.colorscale = None
        self.showscale = None
        if not isinstance(color, str) and color is not None and len(self.color.shape) == 2:
            print("asdas")
            self.colorscale = []
            colors = []
            for indx, c in enumerate(color):
                # ToDo: Error when color is a single element array!
                self.colorscale.append([indx/(len(color)-1), label_rgb(tuple(c))])
                colors.append(indx/(len(color)-1))
            self.color = colors
            self.showscale = False
        
        self.reset(self.intr_cam)
        self.specs = [[{'type': 'scene'}]]
        self.attrs = [[{'has_colorbar': False}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def reset(self, cam):
        """Reset the mesh to the original set of vertices, ignoring any previous 
        intrinsic projections using camera params.
        """
        if cam is not None:
            # xyz2uv is batched (or vectorized on cpu)
            uv = func_xyz2uv()(self.v, cam)
            uvd = np.hstack([uv, self.v[:, 2:]])
        else:
            uvd = self.v
        self.gobjs = (go.Mesh3d(x=uvd[:, 0], y=uvd[:, 1], z=uvd[:, 2], i=self.F[:, 0], j=self.F[:, 1], k=self.F[:, 2],
                              color=self.color, colorscale=self.colorscale, showscale=self.showscale,
                              opacity=self.opacity, flatshading=True, hoverinfo=self.hoverinfo, **self.kwargs),)
        return self

    def update_cam(self, camera):
        self.intr_cam = camera
        self.reset(camera)
        return self
        
class Scatter3D(Plotter):
    """Plotting class for 3D point sets.
    
    Args:
    pts - points
    hoverinfo - whether hovering over points show additional information
    intr_cam - intrinsic camera
    showscale - show color scale if provided
    """
    def __init__(self, pts, size=None, opacity=1, color=None, colorscale=None, showscale=None, text=None, hoverinfo=None, intr_cam=None, **kwargs):
        self.pts, self.size, self.opacity, self.text = pts, size, opacity, text
        self.color, self.colorscale, self.showscale =  color, colorscale, showscale
        self.hoverinfo, self.hovertext, self.intr_cam = hoverinfo, None, intr_cam
        self.kwargs = kwargs
        if text is not None:
            self.hoverinfo = 'text'
            self.hovertext = np.arange(pts.shape[0]) if isinstance(text, int) else text
        
        self.colorscale = None
        if not isinstance(color, str) and color is not None and len(self.color.shape) == 2:
            self.colorscale = []
            colors = []
            for indx, c in enumerate(color):
                # ToDo: Error when color is a single element array!
                self.colorscale.append([indx/(len(color)-1), label_rgb(tuple(c))])
                colors.append(indx/(len(color)-1))
            self.color = colors
            self.showscale = False
        self.reset(self.intr_cam)
        self.specs = [[{'type': 'scene'}]]
        self.attrs = [[{'has_colorbar': False}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def reset(self, cam):
        """Reset the point set to the original set of vertices, ignoring any previous 
        intrinsic projections using camera params.
        """
        if cam is not None:
            # xyz2uv is batched (or vectorized on cpu)
            uv = func_xyz2uv()(self.pts, cam)
            print("uv", uv.shape)
            print("pts", self.pts[:, 2:].shape)
            uvd = np.hstack([uv, self.pts[:, 2:]])
        else:
            uvd = self.pts

        self.gobjs = (go.Scatter3d(x=uvd[:, 0], y=uvd[:, 1], z=uvd[:, 2], mode='markers', marker=dict(size=self.size,
                                   color=self.color, opacity=self.opacity, colorscale=self.colorscale, showscale=self.showscale), hoverinfo=self.hoverinfo,
                                   hovertext=self.hovertext, **self.kwargs),)
        return self
    
    def update_cam(self, camera):
        self.intr_cam = camera
        self.reset(camera)
        return self
        
class Scatter2D(Plotter):
    """Plotting class for 2D point sets.
    
    Args:
    pts - points
    hoverinfo - whether hovering over points show additional information
    mode - use markers or lines for scatter plot
    """
    def __init__(self, pts, size=None, color=None, opacity=1, smoothing=None, text=None, hoverinfo=None, mode="markers", **kwargs):
        self.pts, self.size, self.color, self.opacity, self.text, self.mode= pts, size, color, opacity, text, mode
        self.hoverinfo, self.hovertext = hoverinfo, None
        self.smoothing = smoothing
        self.kwargs = kwargs
        
        if text is not None:
            self.hoverinfo = 'text'
            self.hovertext = np.arange(pts.shape[0]) if isinstance(text, int) else text

        self.reset()
        self.specs = [[{'type': 'xy'}]]
        self.attrs = [[{'has_colorbar': False}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def reset(self):
        """Reset plot to reflect original set of inputs passed during initialisation.
        """
        if self.mode == 'markers':
            self.gobjs = (go.Scatter(x=self.pts[:, 0], y=self.pts[:, 1], mode=self.mode, marker=dict(
            size=self.size, color=self.color, opacity=self.opacity, showscale=False), hoverinfo=self.hoverinfo, hovertext=self.hovertext, 
                                    **self.kwargs),)
        elif self.mode == 'lines':
            self.gobjs = (go.Scatter(x=self.pts[:, 0], y=self.pts[:, 1], mode=self.mode, line=dict(
            color=self.color, smoothing=self.smoothing), hoverinfo=self.hoverinfo, hovertext=self.hovertext, 
                                    **self.kwargs),)
        return self
    
class Wireframe(Plotter):
    """Plotting class for 3D wireframes (or network graphs).
    
    Args:
    v - vertices
    e - edges
    dash - line dash type
    """
    def __init__(self, v, e, width=None, dash=None, color=None, hoverinfo=None, intr_cam=None, **kwargs):
        self.v, self.e, self.width, self.dash, self.color = v, e, width, dash, color
        self.hoverinfo, self.intr_cam = hoverinfo, intr_cam
        self.kwargs = kwargs

        self.colorscale = None
        if not isinstance(color, str) and color is not None:
            self.colorscale = []
            colors = []
            for indx, c in enumerate(color):
                self.colorscale.append([indx/(len(color)-1), label_rgb(tuple(c))])
                colors.append(indx/(len(color)-1))
            self.color = colors
        self.reset(self.intr_cam)
        self.specs = [[{'type': 'scene' if len(self.v[0]) == 3 else 'xy'}]]
        self.attrs = [[{'has_colorbar': False}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def update_cam(self, camera):
        self.intr_cam = camera
        self.reset(camera)
        return self

    def reset(self, cam):
        """Reset the wireframe to the original set of vertices, ignoring any previous 
        intrinsic projections using camera params.
        """
        if cam is not None:
            # xyz2uv is batched (or vectorized on cpu)
            uv = func_xyz2uv()(self.v, cam)
            uvd = np.hstack([uv, self.v[:, 2:]])
        else:
            uvd = self.v
        
        verts_in_edges = uvd[self.e]
        Xl = []
        Yl = []
        Zl = []
        dim = len(self.v[0])
        sides = len(self.e[0])
        for vert_in_edge in verts_in_edges:
            Xl.extend([vert_in_edge[k%sides][0] for k in range(sides+(sides>2))] + [None])
            Yl.extend([vert_in_edge[k%sides][1] for k in range(sides+(sides>2))] + [None])
            if dim == 3:
                Zl.extend([vert_in_edge[k%sides][2] for k in range(sides+(sides>2))] + [None])
        if dim == 3:
            self.gobjs = (go.Scatter3d(x=Xl, y=Yl, z=Zl, mode='lines', line=dict(width=self.width, dash=self.dash, 
              color=self.color, colorscale=self.colorscale, showscale=False), hoverinfo=self.hoverinfo, **self.kwargs),)
        else:
            self.gobjs = (go.Scatter(x=Xl, y=Yl, mode='lines', line=dict(width=self.width, dash=self.dash, 
              color=self.color), hoverinfo=self.hoverinfo, **self.kwargs),)
        return self
    
class PWImage(Plotter):
    """Feed in numpy uint8 image array. 
    Dimensions: (h x w) or (h x w x 3)
    """
    def __init__(self, img, dx=None, dy=None, opacity=None):
        self.img, self.dx, self.dy, self.opacity = img, dx, dy, opacity
        if len(self.img.shape) == 2:
            self.img = self.img[:, :, np.newaxis]
            self.img  = np.concatenate([self.img, self.img, self.img], axis=-1)
        self.reset(dx, dy)
        self.specs = [[{'type': 'image'}]]
        self.attrs = [[{'has_colorbar': False}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def reset(self, dx, dy):
        self.gobjs = (go.Image(z=self.img, dx=self.dx, dy=self.dy, opacity=self.opacity),)
        return self

    def update_size(self, dx=None, dy=None):
        self.reset(dx, dy)
        return self
    
class AreaShade2DLinePlot(Plotter):
    """Feed in 4 arrays - x, y and standard deviation on either side.
    https://plotly.com/python/filled-area-plots/"""
    def __init__(self, x, y, s1, s2, dx=None, dy=None, color=None, width=None, opacity=None):
        self.x, self.y, self.s1, self.s2 = x, y, s1, s2
        self.color, self.opacity, self.width = color, opacity, width
        self.dx, self.dy = dx, dy
        self.reset(dx, dy)
        self.specs = [[{'type': 'xy'}]]
        self.attrs = [[{'has_colorbar': False}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def reset(self, dx, dy):
        self.gobjs = (go.Scatter(x=self.x, y=self.y+self.s2,
                                     mode='lines',
                                     line=dict(color=self.color, width = self.width),
                                     name=''),
                      go.Scatter(x=self.x, y=self.y,
                         mode='lines',
                         line=dict(color=self.color),
                         fill='tonexty',
                         name='training error'),
                      go.Scatter(x=self.x, y=self.y-self.s1,
                         mode='lines',
                         line=dict(color=self.color, width = self.width),
                         fill='tonexty',
                         name='standard deviation'))
        return self
    
class Heatmap(Plotter):
    """Feed in numpy float array for heatmap.
    """
    def __init__(self, hm, dx=None, dy=None, opacity=None):
        self.has_colorbar = True
        self.hm, self.dx, self.dy, self.opacity = hm, dx, dy, opacity
        self.reset(dx, dy)
        self.specs = [[{'type': 'heatmap'}]]
        self.attrs = [[{'has_colorbar': True}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def reset(self, dx, dy):
        self.gobjs = (go.Heatmap(z=self.hm, x=np.arange(len(self.hm)), y=np.arange(len(self.hm)), dx=self.dx, dy=self.dy, opacity=self.opacity),)
        return self

    def update_size(self, dx=None, dy=None):
        self.reset(dx, dy)
        return self
    
    def to_3d(self):
        self.specs = [[{'type': 'surface'}]]
        self.gobjs = (go.Surface(z=self.hm, opacity=self.opacity),)
        self.fig = self.make_subplot()
        return self
    
class PolarPlot(Plotter):
    """Feed in a numpy array.
    """
    def __init__(self, r, theta, dr=None, r0=None, dtheta=None, theta0=None, auto_uniform=False, color=None, colorscale=None, showscale=None, size=None, mode="marker"):
        self.r, self.r0, self.dr = r, r0, dr
        self.color, self.colorscale, self.showscale = color, colorscale, showscale
        self.size = size
        self.theta, self.theta0, self.dtheta = theta, theta0, dtheta
        self.mode = mode
        if auto_uniform:
            # todo
            pass
        self.colorscale = None
        if not isinstance(color, str) and color is not None and len(self.color.shape) == 2:
            self.colorscale = []
            colors = []
            for indx, c in enumerate(color):
                # ToDo: Error when color is a single element array!
                self.colorscale.append([indx/(len(color)-1), label_rgb(tuple(c))])
                colors.append(indx/(len(color)-1))
            self.color = colors
            self.showscale = False
        self.reset()
        self.specs = [[{'type': 'polar'}]]
        self.attrs = [[{'has_colorbar': False}]]
        super().__init__(self.gobjs, self.specs, self.attrs)
        
    def reset(self):
        self.gobjs = (go.Scatterpolar(
                r = self.r,
                r0 = self.r0,
                dr = self.dr,
                theta0 = self.theta0,
                dtheta = self.dtheta,
                theta = self.theta,
                mode = 'markers',
                marker=dict(size=self.size,
                            color=self.color, 
                            colorscale=self.colorscale, 
                            showscale=self.showscale)
            ),)
        return self

        
