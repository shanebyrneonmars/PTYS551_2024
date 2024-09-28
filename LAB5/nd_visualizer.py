import sys
import pandas as pd
from PyQt5.QtWidgets import QMenu, QApplication, QTableWidget, QTableWidgetItem, QMainWindow, QAction, QFileDialog, QWidget, QVBoxLayout, QComboBox, QLabel, QHBoxLayout, QPushButton, QStatusBar, QMessageBox, QSlider, QColorDialog
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np
import json
from copy import deepcopy

class EndmemberManager(QWidget):
    updated_selection = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.endmember_groups = {}

    def init_table(self):
        # Main layout for the widget
        main_layout = QVBoxLayout()

        # Button layout
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Group")
        add_button.clicked.connect(self.add_group)
        button_layout.addWidget(add_button)
        remove_button = QPushButton("Remove Group")
        remove_button.clicked.connect(self.remove_group)
        button_layout.addWidget(remove_button)

        # Table widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Group Name", "Number of Points", "Color"])

        # Enable context menu
        self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.open_context_menu)

        # link clicks
        # link cell clicks to select the working group
        self.table_widget.cellClicked.connect(self.select_working_group)

        # Connect cellChanged signal to update group name
        self.table_widget.cellChanged.connect(self.update_group_name)

        # Populate table
        self.update_table()

        # Add the button layout and table widget to the main layout
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.table_widget)

        # Set the layout for the main widget
        self.setLayout(main_layout)
        self.show()

    def open_context_menu(self, position):
        # Get the index of the right-clicked cell
        index = self.table_widget.indexAt(position)
        if index.isValid():
            row = index.row()
            column = index.column()
            if column == 2:  # Only show context menu for Color column
                menu = QMenu()
                change_color_action = menu.addAction("Change Color")
                action = menu.exec_(self.table_widget.viewport().mapToGlobal(position))
                if action == change_color_action:
                    self.change_color(row)

    def change_color(self, row):
        color_dialog = QColorDialog()
        if color_dialog.exec_():
            color = color_dialog.selectedColor()
            group_name = self.table_widget.item(row, 0).text()
            self.endmember_groups[group_name]['color'] = color
            self.update_table()

    # add a button to add a new group to the table
    def add_group(self):
        # create a new group with a default name
        new_number = 0
        new_group_name = f'Class {new_number}'
        while new_group_name in self.endmember_groups:
            new_number += 1
            new_group_name = f'Class {new_number}'
        self.endmember_groups[new_group_name] = {'indices': [], 'color': QColor(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))}
        self.selected_group = new_group_name
        self.update_table()
        self.updated_selection.emit()

    # add a button to remove a group from the table
    def remove_group(self):
        # remove the selected group from the table
        if self.selected_group:
            self.endmember_groups.pop(self.selected_group)
            self.update_table()
            # update the scatter plot
            self.updated_selection.emit()
        else:
            print("No group selected", file=sys.stderr)

    def select_working_group(self):
        # user selects a working group by clicking on the group in the table
        # the selected group is then used for adding or removing points
        self.selected_group = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        print(f'Working group is {self.selected_group}')

    def add_points_to_group(self, points, group_name):
        if group_name not in self.endmember_groups:
            # Assign a random color for the group
            color = QColor(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            self.endmember_groups[group_name] = {'indices': [], 'color': color}
        self.endmember_groups[group_name]['indices'].extend(points)
        # remove duplicates from endmember group
        self.endmember_groups[group_name]['indices'] = list(set(self.endmember_groups[group_name]['indices']))
        # update points to reflect removal of duplicates
        points = self.endmember_groups[group_name]['indices']
        print(f"Added {len(points)} points to group {group_name}")
        self.update_table()

    def remove_points_from_group(self, points, group_name):
        if group_name in self.endmember_groups:
            for point in points:
                try:
                    self.endmember_groups[group_name]['indices'].remove(point)
                except ValueError:
                    print(f"Point {point} not found in group {group_name}", file=sys.stderr)
            # remove duplicates from endmember group
            self.endmember_groups[group_name]['indices'] = list(set(self.endmember_groups[group_name]['indices']))
            # update points to reflect removal of duplicates
            points = self.endmember_groups[group_name]['indices']
            print(f"Removed {len(points)} points from group {group_name}")
            self.update_table()
        else:
            print(f"Group {group_name} does not exist", file=sys.stderr)

    def delete_group(self, group_name):
        if group_name in self.endmember_groups:
            del self.endmember_groups[group_name]
            print(f"Deleted group {group_name}")
        else:
            print(f"Group {group_name} does not exist", file=sys.stderr)

    def update_group_name(self, old_name, new_name):
        if old_name in self.endmember_groups:
            self.endmember_groups[new_name] = self.endmember_groups.pop(old_name)
            print(f"Renamed group from {old_name} to {new_name}")
        else:
            print(f"Group {old_name} does not exist", file=sys.stderr)

    def update_group_name(self, row, column):
        if column == 0:  # Only update if the Group Name column is changed
            old_group_name = list(self.endmember_groups.keys())[row]
            new_group_name = self.table_widget.item(row, column).text()
            if new_group_name != old_group_name:
                self.endmember_groups[new_group_name] = self.endmember_groups.pop(old_group_name)
                print(f"Updated group name from '{old_group_name}' to '{new_group_name}'")


    def update_table(self):
        self.table_widget.clearContents()
        self.table_widget.setRowCount(len(self.endmember_groups))
        for i, (group_name, details) in enumerate(self.endmember_groups.items()):
            self.table_widget.setItem(i, 0, QTableWidgetItem(group_name))
            self.table_widget.setItem(i, 1, QTableWidgetItem(str(len(details['indices']))))
            # Assuming the color is stored as a QColor object
            color_item = QTableWidgetItem()
            color_item.setBackground(details['color'])
            self.table_widget.setItem(i, 2, color_item)
        print("Endmember manager table updated")
        self.updated_selection.emit()

    def handle_new_selection(self, points, group_name, addPoints):
        if addPoints:
            try:
                self.add_points_to_group(points, group_name)
                print(f"Handled new selection for group {group_name}")
            except Exception as e:
                print(f"Error handling new selection: {e}", file=sys.stderr)
        else:
            try:
                self.remove_points_from_group(points, group_name)
                print(f"Handled new selection for group {group_name}")
            except Exception as e:
                print(f"Error handling new selection: {e}", file=sys.stderr)

class SelectFromCollection(QObject):
    updated_fc = pyqtSignal(np.ndarray)

    def __init__(self, ax, collection, endmember_manager, addPoints, alpha_other=0.3):
        super().__init__()
        self.endmember_manager = endmember_manager
        self.endmember_manager.updated_selection.connect(self.update_scatter)
        self.addPoints = addPoints
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor set')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def update_scatter(self):
        self.fc[:, -1] = self.alpha_other  # Dim unselected points
        self.fc[:, :-1] = [0, 0, 0] # set points to default black color
        # set face colors based on the endmember_group colors
        for _, details in self.endmember_manager.endmember_groups.items():
            if len(details['indices']) != 0:
                indices = details['indices']
                color = details['color']
                color_ = [c/255 for c in color.getRgb()[:3]]
                self.fc[indices, :-1] = color_
                self.fc[indices, -1] = 1
        
        self.updated_fc.emit(self.fc)
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0] 

        # create a new group with a default name
        new_number = 0
        new_group_name = f'Class {new_number}'
        while new_group_name in self.endmember_manager.endmember_groups:
            new_number += 1
            new_group_name = f'Class {new_number}'

        working_group = self.endmember_manager.selected_group if hasattr(self.endmember_manager, 'selected_group') else new_group_name
        self.endmember_manager.handle_new_selection(self.ind.tolist(), working_group, self.addPoints)  
        self.update_scatter()
        self.updated_fc.emit(self.fc)

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1  # Reset the facecolor alpha of all points
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = 'ND-Visualizer'
        self.left = 10
        self.top = 10
        self.width = 600
        self.height = 1000

        self.dataset = None
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.dimensionSelectors = []
        self.animation = None
        self.currentFrame = 0
        self.animationDirection = 1
        self.selector = None
        self.lassoEnabled = False
        self.endmember_manager = EndmemberManager()
        self.endmember_manager.init_table()
        self.fc = None
        self.original_handlers = {}
        # self.endmember_manager.updated_selection.connect(self.slot_update_plot)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        mainLayout = QVBoxLayout()
        plotLayout = QVBoxLayout()
        dimensionSelectionLayout = QHBoxLayout()
        animationLayout = QHBoxLayout()
        lassoLayout = QHBoxLayout()

        for _ in range(3):
            comboBox = QComboBox()
            comboBox.currentIndexChanged.connect(self.updatePlot)
            self.dimensionSelectors.append(comboBox)
            dimensionSelectionLayout.addWidget(comboBox)

        self.backwardAnimationButton = QPushButton('<')
        self.startAnimationButton = QPushButton('Play')
        self.stopAnimationButton = QPushButton('Stop')
        self.forwardAnimationButton = QPushButton('>')
        animationLayout.addWidget(self.backwardAnimationButton)
        animationLayout.addWidget(self.startAnimationButton)
        animationLayout.addWidget(self.stopAnimationButton)
        animationLayout.addWidget(self.forwardAnimationButton)
        self.backwardAnimationButton.clicked.connect(self.backwardAnimation)
        self.startAnimationButton.clicked.connect(self.startStopAnimation)
        self.stopAnimationButton.clicked.connect(self.stopAnimation)
        self.forwardAnimationButton.clicked.connect(self.forwardAnimation)

        self.lassoSelectButton = QPushButton('Lasso Select Add')
        self.lassoSelectButton.clicked.connect(self.lassoSelectOn)
        lassoLayout.addWidget(self.lassoSelectButton)

        self.lassoDeselectButton = QPushButton('Lasso Select Subtract')
        self.lassoDeselectButton.clicked.connect(self.lassoDeselectOn)
        lassoLayout.addWidget(self.lassoDeselectButton)

        self.lassoSelectOffButton = QPushButton('Lasso Select Off')
        self.lassoSelectOffButton.clicked.connect(self.lassoSelectOff)
        lassoLayout.addWidget(self.lassoSelectOffButton)

        self.pointSizeSlider = QSlider(Qt.Horizontal)
        self.pointSizeSlider.setMinimum(1)
        self.pointSizeSlider.setMaximum(10)
        self.pointSizeSlider.setValue(1)
        self.pointSizeSlider.setTickPosition(QSlider.TicksBelow)
        self.pointSizeSlider.setTickInterval(1)
        self.pointSizeSlider.valueChanged.connect(self.updatePointSize)
        mainLayout.addWidget(self.pointSizeSlider)

        plotLayout.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        plotLayout.addWidget(self.toolbar)

        mainLayout.addLayout(dimensionSelectionLayout)
        mainLayout.addLayout(animationLayout)
        mainLayout.addLayout(lassoLayout)
        mainLayout.addLayout(plotLayout)
        
        self.central_widget = QWidget()
        self.central_widget.setLayout(mainLayout)
        self.setCentralWidget(self.central_widget)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu('File')

        open_action = QAction('Open', self)
        open_action.triggered.connect(self.openFileNameDialog)
        file_menu.addAction(open_action)

        export_action = QAction('Export Selected Points', self)
        export_action.triggered.connect(self.exportSelectedPoints)
        file_menu.addAction(export_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        endmember_menu = self.menu_bar.addMenu('Manage Classes')
        open_endmember_manager_action = QAction('Open Endmember Manager', self)
        open_endmember_manager_action.triggered.connect(self.openEndmemberManager)
        endmember_menu.addAction(open_endmember_manager_action)

    def openEndmemberManager(self):
        # Ensure the endmember_manager is properly initialized and updated
        if not hasattr(self, 'endmember_manager'):
            self.endmember_manager = EndmemberManager()
        # initiate the endmember manager UI
        self.endmember_manager.init_table()

    def update_fc(self, fc):
        print(f'received face colors')
        self.fc = fc

    def lassoDeselectOn(self):
        self.lassoEnabled = True
        self.lassoAddPoints = False
        self.handleLassoSelect()

    def lassoSelectOn(self):
        self.lassoEnabled = True
        self.lassoAddPoints = True
        self.handleLassoSelect()

    def lassoSelectOff(self):
        self.lassoEnabled = False
        self.handleLassoSelect()
    
    def _on_event(self, event):
        pass

    def handleLassoSelect(self):
        
        if self.lassoEnabled:
            self.statusBar.showMessage("Lasso Select Enabled")
            self.toolbar.setEnabled(False)

            self.ax.mouse_init(rotate_btn=None, pan_btn=None, zoom_btn=None)

            if self.selector:
                self.selector.disconnect()
            # logic to initialize the lasso selector
            collection = self.scatter
            self.selector = SelectFromCollection(self.ax, collection, self.endmember_manager, self.lassoAddPoints)
            self.selector.updated_fc.connect(self.update_fc)

        else:
            self.statusBar.showMessage("Lasso Select Disabled")
            self.toolbar.setEnabled(True)

            self.ax.mouse_init(rotate_btn=1, pan_btn=3, zoom_btn=2)

            if self.selector:
                self.selector.disconnect()
                self.selector = None

    def openFileNameDialog(self):
        try:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv)", options=options)
            if fileName:
                self.dataset = pd.read_csv(fileName)
                self.updateDimensionSelectors()
                self.updatePlot()
        except Exception as e:
            self.statusBar.showMessage(f"Error opening file: {e}")

    def updateDimensionSelectors(self):
        columnNames = self.dataset.columns.tolist()
        for i, comboBox in enumerate(self.dimensionSelectors):
            comboBox.clear()
            comboBox.addItems(columnNames)
            if i < len(self.dimensionSelectors):
                comboBox.setCurrentIndex(i)

    def slot_update_plot(self):
        self.scatter.set_facecolors(self.fc)
        self.ax.figure.canvas.draw_idle()

    def updatePlot(self, dimensions=None, fc=None):
        if isinstance(dimensions, int):
            dimensions = [comboBox.currentIndex() for comboBox in self.dimensionSelectors]
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        if self.dataset is not None:
            dims = dimensions if dimensions is not None else [comboBox.currentIndex() for comboBox in self.dimensionSelectors]

            if len(dims) < 3:
                self.statusBar.showMessage("Please select three dimensions to plot.")
                return
            else:
                pointSize = self.pointSizeSlider.value()
                self.scatter = ax.scatter(self.dataset.iloc[:, dims[0]], self.dataset.iloc[:, dims[1]], self.dataset.iloc[:, dims[2]], s=pointSize**2)
                ax.set_xlabel(self.dataset.columns[dims[0]])
                ax.set_ylabel(self.dataset.columns[dims[1]])
                ax.set_zlabel(self.dataset.columns[dims[2]])
                self.ax = ax
                if fc is not None:
                    self.scatter.set_facecolors(fc)
                    self.canvas.draw_idle()
                else:
                    fc = self.scatter.get_facecolors()
                    fc[:, :-1] = [0, 0, 0] # set points to default black color
                    self.scatter.set_facecolors(fc)
        
        self.canvas.draw_idle()

    def updatePointSize(self):
        self.updatePlot()

    def startStopAnimation(self):
        if self.animation is None:
            self.startAnimation()
        else:
            self.stopAnimation()

    def startAnimation(self):
        if self.dataset is not None:
            def update(frame):
                self.currentFrame += self.animationDirection
                dims = [comboBox.currentIndex()+1 for comboBox in self.dimensionSelectors]
                if all(dim < self.dataset.shape[1] for dim in dims):
                    self.updatePlot(dimensions=dims, fc=self.fc)
                    for i, comboBox in enumerate(self.dimensionSelectors):
                        comboBox.setCurrentIndex(dims[i])
                else:
                    self.statusBar.showMessage("Dataset does not have enough dimensions for animation.")
                    self.stopAnimation()
                    return
            
            self.animation = FuncAnimation(self.figure, update, frames=np.arange(0, self.dataset.shape[1]), interval=100, repeat=True)
            self.canvas.draw()

    def stopAnimation(self):
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
            self.currentFrame = 0
            self.updatePlot(fc=self.fc)

    def backwardAnimation(self):
        if self.dataset is not None:
            dims = [comboBox.currentIndex()-1 for comboBox in self.dimensionSelectors]
            if all(dim >= 0 for dim in dims):
                for i, comboBox in enumerate(self.dimensionSelectors):
                    comboBox.setCurrentIndex(dims[i])
                    
                self.updatePlot(dimensions=dims, fc=self.fc)
            else:
                self.statusBar.showMessage("Dataset does not have enough dimensions for animation.")

    def forwardAnimation(self):
        if self.dataset is not None:
            dims = [comboBox.currentIndex()+1 for comboBox in self.dimensionSelectors]
            if all(dim < self.dataset.shape[1] for dim in dims):
                for i, comboBox in enumerate(self.dimensionSelectors):
                    comboBox.setCurrentIndex(dims[i])
                
                self.updatePlot(dimensions=dims, fc=self.fc)
                # if hasattr(self, 'selector'):
                    # update the selector for new dimensions
                    # collection = self.scatter
                    # self.selector = SelectFromCollection(self.ax, collection, self.endmember_manager)
            else:
                self.statusBar.showMessage("Dataset does not have enough dimensions for animation.")

    def exportSelectedPoints(self):
        if hasattr(self.endmember_manager, 'endmember_groups'):
            
            save_file = deepcopy(self.endmember_manager.endmember_groups)
            # convert Qcolor to float color
            for group_name, details in save_file.items():
                color = details['color']
                color = [c/255 for c in color.getRgb()[:3]]
                save_file[group_name]['color'] = color

            filename, _ = QFileDialog.getSaveFileName(self, "Save Selected Points", "", "json Files (*.json)")

            if filename:
                #  Save to JSON file
                with open(filename, 'w') as file:
                    json.dump(save_file, file, indent=4)
                # # Save to pickle file
                # with open(filename, 'wb') as file:
                #     pickle.dump(self.endmember_manager.endmember_groups, file)

                # selected_data.to_csv(filename, index=False)
                QMessageBox.information(self, "Export Successful", f"Selected points have been exported to {filename}")
        else:
            QMessageBox.warning(self, "No Selection", "No points have been selected for export.")

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()